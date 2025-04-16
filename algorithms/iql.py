from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
import os
import warnings

import distrax
import d4rl
import flax.linen as nn
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "iql"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimization ---
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    # --- IQL ---
    beta: float = 3.0
    iql_tau: float = 0.7
    exp_adv_clip: float = 100.0


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "actor dual_q dual_q_target value")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=2,  # Two Q networks
        )
        q_values = vmap_critic(self.obs_mean, self.obs_std)(obs, action)
        return q_values


class StateValueFunction(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)


class TanhGaussianActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    log_std_max: float = 2.0
    log_std_min: float = -20.0

    @nn.compact
    def __call__(self, x, eval=False):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.tanh(x)
        if eval:
            return distrax.Deterministic(x)
        logstd = self.param(
            "logstd",
            init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
        )
        std = jnp.exp(jnp.clip(logstd, self.log_std_min, self.log_std_max))
        return distrax.Normal(x, std)


def create_train_state(args, rng, network, dummy_input):
    lr_schedule = optax.cosine_decay_schedule(args.lr, args.num_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr_schedule, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)
    obs = env.reset()

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(onp.array(action))

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(args, actor_apply_fn, q_apply_fn, value_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.dual_q.params,
            agent_state.dual_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.dual_q_target.replace(
            step=agent_state.dual_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(dual_q_target=updated_q_target)

        # --- Compute targets ---
        v_target = q_apply_fn(agent_state.dual_q_target.params, batch.obs, batch.action)
        v_target = v_target.min(-1)
        next_v_target = value_apply_fn(agent_state.value.params, batch.next_obs)
        q_targets = batch.reward + args.gamma * (1 - batch.done) * next_v_target

        # --- Update Q and value functions ---
        def _q_loss_fn(params):
            # Compute loss for both critics
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            q_loss = jnp.square(q_pred - jnp.expand_dims(q_targets, axis=-1)).mean()
            return q_loss

        @partial(jax.value_and_grad, has_aux=True)
        def _value_loss_fn(params):
            adv = v_target - value_apply_fn(params, batch.obs)
            # Asymmetric L2 loss
            value_loss = jnp.abs(args.iql_tau - (adv < 0.0).astype(float)) * (adv**2)
            return jnp.mean(value_loss), adv

        q_loss, q_grad = jax.value_and_grad(_q_loss_fn)(agent_state.dual_q.params)
        (value_loss, adv), value_grad = _value_loss_fn(agent_state.value.params)
        agent_state = agent_state._replace(
            dual_q=agent_state.dual_q.apply_gradients(grads=q_grad),
            value=agent_state.value.apply_gradients(grads=value_grad),
        )

        # --- Update actor ---
        exp_adv = jnp.exp(adv * args.beta).clip(max=args.exp_adv_clip)

        @jax.value_and_grad
        def _actor_loss_function(params):
            def _compute_loss(transition, exp_adv):
                pi = actor_apply_fn(params, transition.obs)
                bc_loss = -pi.log_prob(transition.action)
                return exp_adv * bc_loss.sum()

            actor_loss = jax.vmap(_compute_loss)(batch, exp_adv)
            return actor_loss.mean()

        actor_loss, actor_grad = _actor_loss_function(agent_state.actor.params)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        loss = {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
        return (rng, agent_state), loss

    return _train_step


if __name__ == "__main__":
    # --- Parse arguments ---
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    # --- Initialize logger ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_agent",
        )

    # --- Initialize environment and dataset ---
    env = gym.vector.make(args.dataset, num_envs=args.eval_workers)
    dataset = d4rl.qlearning_dataset(gym.make(args.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    # --- Initialize agent and value networks ---
    num_actions = env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_net = TanhGaussianActor(num_actions, obs_mean, obs_std)
    q_net = DualQNetwork(obs_mean, obs_std)
    value_net = StateValueFunction(obs_mean, obs_std)

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q, rng_value = jax.random.split(rng, 4)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        value=create_train_state(args, rng_value, value_net, [dummy_obs]),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, actor_net.apply, q_net.apply, value_net.apply, dataset
    )

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores = d4rl.get_normalized_score(args.dataset, returns) * 100.0

        # --- Log metrics ---
        step = (eval_idx + 1) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        if args.log:
            log_dict = {
                "return": returns.mean(),
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets = onp.concat([eval_agent(args, _rng, env, agent_state) for _rng in _rng])
        scores = d4rl.get_normalized_score(args.dataset, rets) * 100.0
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        # --- Write final returns to file ---
        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{args.algorithm}_{args.dataset}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    if args.log:
        wandb.finish()
