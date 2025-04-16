from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import warnings

import distrax
import d4rl
import flax.linen as nn
from flax.linen.initializers import constant, uniform
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
    algorithm: str = "rebrac"
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
    lr: float = 1e-3
    batch_size: int = 1024
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    # --- TD3+BC ---
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    num_critic_updates_per_step: int = 2
    # --- REBRAC ---
    critic_bc_coef: float = 0.01
    actor_bc_coef: float = 0.001
    actor_ln: bool = False
    critic_ln: bool = True
    norm_obs: bool = False


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple(
    "AgentTrainState", "actor actor_target dual_q dual_q_target"
)
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

    @nn.compact
    def __call__(self, obs, action):
        if self.norm_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.use_ln else x
        q = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return q.squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

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
        q_fn = vmap_critic(self.obs_mean, self.obs_std, self.use_ln, self.norm_obs)
        q_values = q_fn(obs, action)
        return q_values


class DeterministicTanhActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

    @nn.compact
    def __call__(self, x):
        if self.norm_obs:
            x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.use_ln else x
        init_fn = sym(1e-3)
        action = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)(x)
        pi = distrax.Transformed(
            distrax.Deterministic(action),
            distrax.Tanh(),
        )
        return pi


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(args.lr, eps=1e-5),
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
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs)
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


def make_train_step(args, actor_apply_fn, q_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update critics ---
        def _update_critics(runner_state, _):
            rng, agent_state = runner_state

            def _compute_target(rng, transition):
                next_obs = transition.next_obs

                # --- Sample noised action ---
                next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
                rng, rng_action, rng_noise = jax.random.split(rng, 3)
                action = next_pi.sample(seed=rng_action)
                noise = jax.random.normal(rng_noise, shape=action.shape)
                noise *= args.policy_noise
                noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
                action = jnp.clip(action + noise, -1, 1)
                bc_loss = jnp.square(action - transition.next_action).sum()

                # --- Compute targets ---
                target_q = q_apply_fn(
                    agent_state.dual_q_target.params, next_obs, action
                )
                next_q_value = jnp.min(target_q) - args.critic_bc_coef * bc_loss
                next_q_value = (1.0 - transition.done) * next_q_value
                return transition.reward + args.gamma * next_q_value, bc_loss

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            target_fn = jax.vmap(_compute_target)
            targets, bc_loss = target_fn(rng_targets, batch)

            # --- Compute critic loss ---
            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                q_loss = jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).sum(-1)
                return q_loss.mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            updated_q_state = agent_state.dual_q.apply_gradients(grads=q_grad)
            agent_state = agent_state._replace(dual_q=updated_q_state)
            return (rng, agent_state), (q_loss, bc_loss)

        # --- Iterate critic update ---
        (rng, agent_state), (q_loss, critic_bc_loss) = jax.lax.scan(
            _update_critics,
            (rng, agent_state),
            None,
            length=args.num_critic_updates_per_step,
        )

        # --- Update actor ---
        def _actor_loss_function(params):
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
                bc_loss = jnp.square(pi_action - transition.action).sum()
                return q.min(), bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch)
            lambda_ = 1.0 / (jnp.abs(q).mean() + 1e-7)
            lambda_ = jax.lax.stop_gradient(lambda_)
            actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * bc_loss.mean()
            return actor_loss.mean(), (q.mean(), lambda_.mean(), bc_loss.mean())

        loss_fn = jax.value_and_grad(_actor_loss_function, has_aux=True)
        (actor_loss, (q_mean, lambda_, bc_loss)), actor_grad = loss_fn(
            agent_state.actor.params
        )
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        # --- Update target networks ---
        def _update_target(state, target_state):
            new_target_params = optax.incremental_update(
                state.params, target_state.params, args.polyak_step_size
            )
            return target_state.replace(
                step=target_state.step + 1, params=new_target_params
            )

        agent_state = agent_state._replace(
            actor_target=_update_target(agent_state.actor, agent_state.actor_target),
            dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target),
        )

        loss = {
            "actor_loss": actor_loss,
            "q_loss": q_loss.mean(),
            "q_mean": q_mean,
            "lambda": lambda_,
            "bc_loss": bc_loss,
            "critic_bc_loss": critic_bc_loss.mean(),
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
        next_action=jnp.roll(dataset["actions"], -1, axis=0),
        done=jnp.array(dataset["terminals"]),
    )

    # --- Initialize agent and value networks ---
    num_actions = env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_cls = DeterministicTanhActor
    actor_net = actor_cls(num_actions, obs_mean, obs_std, args.actor_ln, args.norm_obs)
    q_net = DualQNetwork(obs_mean, obs_std, args.critic_ln, args.norm_obs)

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q = jax.random.split(rng, 3)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        actor_target=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(args, actor_net.apply, q_net.apply, dataset)

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
