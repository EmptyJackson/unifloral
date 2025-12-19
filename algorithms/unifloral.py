from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
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

from dynamics import (
    Transition,
    load_dynamics_model,
    EnsembleDynamics,  # required for loading dynamics model
    EnsembleDynamicsModel,  # required for loading dynamics model
)

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "unifloral"
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
    actor_lr: float = 1e-3
    lr_schedule: str = "constant"  # ["constant", "cosine"]
    batch_size: int = 1024
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    norm_obs: bool = True
    # --- Actor architecture ---
    actor_num_layers: int = 3
    actor_layer_width: int = 256
    actor_ln: bool = True
    deterministic: bool = True
    deterministic_eval: bool = False
    use_tanh_mean: bool = True
    use_log_std_param: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # --- Critic + value function architecture ---
    num_critics: int = 2
    critic_num_layers: int = 3
    critic_layer_width: int = 256
    critic_ln: bool = True
    # --- Actor loss components ---
    actor_bc_coef: float = 0.001  # Behavior cloning coefficient
    actor_q_coef: float = 1.0  # Q-loss coefficient
    use_q_target_in_actor: bool = False  # Whether to use target Q network for actor
    normalize_q_loss: bool = True  # Whether to normalize Q values in actor loss
    aggregate_q: str = "min"  # ["min", "mean", "first"] - How to aggregate Q values
    # --- AWR (Advantage Weighted Regression) actor ---
    use_awr: bool = False  # Whether to use AWR policy updates
    awr_temperature: float = 1.0  # Temperature for AWR advantages
    awr_exp_adv_clip: float = 100.0  # Maximum exponentiated advantage for AWR
    # --- Critic loss components ---
    num_critic_updates_per_step: int = 2  # Number of critic updates per actor update
    critic_bc_coef: float = 0.01  # Behavior cloning coefficient
    diversity_coef: float = 0.0  # Critic diversity coefficient
    policy_noise: float = 0.2  # Noise added to target actions
    noise_clip: float = 0.5  # Target policy noise limits
    use_target_actor: bool = True  # Whether to use actor target network
    # --- Value function ---
    use_value_target: bool = False  # Whether to use separate value network for targets
    value_expectile: float = 0.7  # Expectile regression coefficient
    # --- Entropy loss ---
    use_entropy_loss: bool = False  # Whether to use SAC entropy regularization
    ent_coef_init: float = 1.0  # Initial entropy coefficient
    actor_entropy_coef: float = 0.0  # Actor entropy coefficient
    critic_entropy_coef: float = 0.0  # Critic entropy coefficient
    # --- World model ---
    model_path: str = ""
    dataset_sample_ratio: float = 1.0  # Dataset sample ratio (set to 1 for model-free)
    rollout_interval: int = 1000
    rollout_length: int = 5
    rollout_batch_size: int = 50000
    model_retain_epochs: int = 5
    step_penalty_coef: float = 0.5


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
    "AgentTrainState", "actor actor_target vec_q vec_q_target value alpha"
)


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    args: Args
    data_mean: Transition
    data_std: Transition

    @nn.compact
    def __call__(self, obs, action):
        if self.args.norm_obs:
            obs = (obs - self.data_mean.obs) / (self.data_std.obs + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(self.args.critic_num_layers):
            x = nn.Dense(self.args.critic_layer_width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.critic_ln else x
        q = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return q.squeeze(-1)


class VectorQ(nn.Module):
    args: Args
    data_mean: Transition
    data_std: Transition

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.args.num_critics,
        )
        q_fn = vmap_critic(self.args, self.data_mean, self.data_std)
        q_values = q_fn(obs, action)
        return q_values


class StateValueFunction(nn.Module):
    args: Args
    data_mean: Transition
    data_std: Transition

    @nn.compact
    def __call__(self, x):
        if self.args.norm_obs:
            x = (x - self.data_mean.obs) / (self.data_std.obs + 1e-3)
        for _ in range(self.args.critic_num_layers):
            x = nn.Dense(self.args.critic_layer_width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.critic_ln else x
        v = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return v.squeeze(-1)


class Actor(nn.Module):
    args: Args
    data_mean: Transition
    data_std: Transition
    num_actions: int

    @nn.compact
    def __call__(self, x, eval=False):
        # --- Compute forward pass ---
        if self.args.norm_obs:
            x = (x - self.data_mean.obs) / (self.data_std.obs + 1e-3)
        for _ in range(self.args.actor_num_layers):
            x = nn.Dense(self.args.actor_layer_width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.actor_ln else x

        # --- Compute mean ---
        init_fn = sym(1e-3)
        mean = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)(x)
        if self.args.use_tanh_mean:
            mean = jnp.tanh(mean)
        if self.args.deterministic or (self.args.deterministic_eval and eval):
            assert self.args.use_tanh_mean, "Deterministic actor requires clipped mean"
            return distrax.Deterministic(mean)

        # --- Compute variance ---
        if self.args.use_log_std_param:
            log_std = self.param(
                "log_std",
                init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
            )
        else:
            std_fn = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)
            log_std = std_fn(x)
        std = jnp.exp(jnp.clip(log_std, self.args.log_std_min, self.args.log_std_max))
        pi = distrax.Normal(mean, std)
        if not self.args.use_tanh_mean:
            pi = distrax.Transformed(pi, distrax.Tanh())
        return pi


class EntropyCoef(nn.Module):
    args: Args

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.args.ent_coef_init)),
        )
        return log_ent_coef


def create_train_state(args, rng, network, dummy_input, is_actor=False, steps=None):
    lr = args.actor_lr if is_actor else args.lr
    if args.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(lr, steps or args.num_updates)
    elif args.lr_schedule != "constant":
        raise ValueError(f"Invalid learning rate schedule: {args.lr_schedule}")
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr, eps=1e-5),
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


def sample_from_buffer(buffer, batch_size, rng):
    """Sample a batch from the buffer."""
    idxs = jax.random.randint(rng, (batch_size,), 0, len(buffer.obs))
    return jax.tree_map(lambda x: x[idxs], buffer)


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


def make_train_step(
    args,
    actor_apply_fn,
    q_apply_fn,
    value_apply_fn,
    alpha_apply_fn,
    dataset,
    rollout_fn,
):
    """Make JIT-compatible agent train step, with optional model-based rollouts."""

    def _train_step(runner_state, _):
        rng, agent_state, rollout_buffer = runner_state

        # --- Update model buffer ---
        if args.dataset_sample_ratio < 1.0:
            params = agent_state.actor.params
            policy_fn = lambda obs, rng: actor_apply_fn(params, obs).sample(seed=rng)
            rng, rng_buffer = jax.random.split(rng)
            rollout_buffer = jax.lax.cond(
                agent_state.actor.step % args.rollout_interval == 0,
                lambda: rollout_fn(rng_buffer, policy_fn, rollout_buffer),
                lambda: rollout_buffer,
            )

        # --- Sample batch ---
        rng, rng_dataset, rng_roll = jax.random.split(rng, 3)
        dataset_size = int(args.batch_size * args.dataset_sample_ratio)
        batch = sample_from_buffer(dataset, dataset_size, rng_dataset)
        if args.dataset_sample_ratio < 1.0:
            rollout_size = args.batch_size - dataset_size
            rollout_batch = sample_from_buffer(rollout_buffer, rollout_size, rng_roll)
            batch = jax.tree_map(
                lambda x, y: jnp.concatenate([x, y]), batch, rollout_batch
            )
        losses = {}

        # --- Update entropy coefficient ---
        if args.use_entropy_loss:
            # --- Compute entropy ---
            pi = jax.vmap(lambda obs: actor_apply_fn(agent_state.actor.params, obs))
            pi_rng, rng = jax.random.split(rng)
            _, log_pi = pi(batch.obs).sample_and_log_prob(seed=pi_rng)
            entropy = -log_pi.sum(-1).mean()

            # --- Compute alpha loss ---
            target_entropy = -batch.action.shape[-1]
            ent_diff = entropy - target_entropy
            alpha_loss_fn = jax.value_and_grad(lambda p: alpha_apply_fn(p) * ent_diff)
            alpha_loss, alpha_grad = alpha_loss_fn(agent_state.alpha.params)
            updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
            agent_state = agent_state._replace(alpha=updated_alpha)
            alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))
            losses.update({"alpha_loss": alpha_loss, "alpha": alpha})

        # --- Update actor ---
        @partial(jax.grad, has_aux=True)
        def _actor_loss_function(params, rng):
            require_bc = args.actor_bc_coef > 0.0 or args.use_awr
            require_q = args.actor_q_coef > 0.0
            if args.use_q_target_in_actor:
                q_params = agent_state.vec_q_target.params
            else:
                q_params = agent_state.vec_q.params

            def _compute_losses(rng, transition):
                # --- Sample action ---
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                losses = {"entropy_loss": log_pi.sum()}

                # --- Compute BC loss ---
                if require_bc:
                    if args.deterministic:
                        bc_loss = jnp.square(sampled_action - transition.action).sum()
                    else:
                        bc_loss = -pi.log_prob(transition.action).sum()

                    # Weight BC loss using either AWR or fixed coefficient
                    bc_coef = args.actor_bc_coef
                    if args.use_awr:
                        bc_q = q_apply_fn(q_params, transition.obs, transition.action)
                        bc_v = value_apply_fn(agent_state.value.params, transition.obs)
                        adv = bc_q.min() - bc_v
                        losses.update({"bc_q": bc_q, "bc_v": bc_v, "adv": adv})
                        adv_max = args.awr_exp_adv_clip
                        aw_coef = jnp.exp(adv * args.awr_temperature).clip(max=adv_max)
                        bc_coef *= aw_coef
                    losses["bc_loss"] = bc_loss
                    losses["bc_coef"] = bc_coef

                # --- Compute Q loss ---
                if require_q:
                    actor_q = q_apply_fn(q_params, transition.obs, sampled_action)
                    if args.aggregate_q == "min":
                        actor_q = jnp.min(actor_q)
                    elif args.aggregate_q == "mean":
                        actor_q = jnp.mean(actor_q)
                    elif args.aggregate_q == "first":
                        actor_q = actor_q[0]
                    else:
                        raise ValueError(f"Unknown Q aggregation: {args.aggregate_q}")
                    losses["q_loss"] = -actor_q
                    losses["actor_q"] = actor_q  # Return for loss normalization
                return losses

            rng = jax.random.split(rng, args.batch_size)
            losses = jax.vmap(_compute_losses)(rng, batch)

            # --- Aggregate losses ---
            losses["actor_loss"] = jnp.zeros(batch.obs.shape[0])
            if require_bc:
                losses["actor_loss"] += losses["bc_loss"] * losses["bc_coef"]
            if require_q:
                q_coef = args.actor_q_coef
                if args.normalize_q_loss:
                    losses["abs_actor_q"] = jnp.abs(losses["actor_q"])
                    lambda_ = 1.0 / (losses["abs_actor_q"].mean() + 1e-7)
                    lambda_ = jax.lax.stop_gradient(lambda_)
                    q_coef *= lambda_
                losses["actor_loss"] += q_coef * losses["q_loss"]
            if args.use_entropy_loss:
                ent_coef = args.actor_entropy_coef * alpha
                losses["actor_loss"] += ent_coef * losses["entropy_loss"]
            losses = jax.tree_map(jnp.mean, losses)
            return losses["actor_loss"], losses

        rng, rng_actor = jax.random.split(rng)
        actor_params = agent_state.actor.params
        actor_grad, actor_losses = _actor_loss_function(actor_params, rng_actor)
        losses.update(actor_losses)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Update critics ---
        def _update_critics(runner_state, _):
            rng, agent_state = runner_state

            def _compute_target(rng, transition):
                # --- Sample noised action ---
                next_obs = transition.next_obs
                if args.use_target_actor:
                    next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
                else:
                    next_pi = actor_apply_fn(agent_state.actor.params, next_obs)
                rng, rng_action, rng_noise = jax.random.split(rng, 3)
                next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng_action)
                noise = jax.random.normal(rng_noise, shape=next_action.shape)
                noise *= args.policy_noise
                noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
                next_action = jnp.clip(next_action + noise, -1, 1)

                # --- Compute target ---
                if args.use_value_target:
                    next_v = value_apply_fn(agent_state.value.params, next_obs)
                else:
                    q_target_params = agent_state.vec_q_target.params
                    next_v = q_apply_fn(q_target_params, next_obs, next_action).min()
                losses = {"critic_next_v": next_v}
                if args.use_entropy_loss:
                    next_v += args.critic_entropy_coef * alpha * log_next_pi.sum()
                    losses["critic_entropy_loss"] = log_next_pi.sum()
                if args.critic_bc_coef > 0.0:
                    bc_loss = jnp.square(next_action - transition.next_action).sum()
                    losses["critic_bc_loss"] = bc_loss
                    next_v -= args.critic_bc_coef * bc_loss
                next_v *= (1.0 - transition.done) * args.gamma
                return transition.reward + next_v, losses

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            targets, critic_losses = jax.vmap(_compute_target)(rng_targets, batch)

            # --- Update critics ---
            @partial(jax.grad, has_aux=True)
            def _q_loss_fn(params):
                def _diversity_loss_fn(obs, action):
                    jac = jax.jacrev(q_apply_fn, argnums=2)(params, obs, action)
                    jac /= jnp.linalg.norm(jac, axis=-1, keepdims=True) + 1e-6
                    div_loss = jac @ jac.T
                    div_loss *= 1.0 - jnp.eye(args.num_critics)
                    return div_loss.sum()

                q_pred = q_apply_fn(params, batch.obs, batch.action)
                q_diff = q_pred - targets.reshape(args.batch_size, 1)
                losses = {"critic_loss": jnp.square(q_diff).sum(-1).mean()}
                if args.diversity_coef > 0.0:
                    batch_div_fn = jax.vmap(_diversity_loss_fn)
                    diversity_loss = batch_div_fn(batch.obs, batch.action)
                    diversity_loss = diversity_loss.mean() / (args.num_critics - 1)
                    losses["diversity_loss"] = diversity_loss
                    losses["critic_loss"] += args.diversity_coef * diversity_loss
                return losses["critic_loss"], losses

            critic_grad, q_losses = _q_loss_fn(agent_state.vec_q.params)
            critic_losses.update(q_losses)
            updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
            agent_state = agent_state._replace(vec_q=updated_q)
            return (rng, agent_state), critic_losses

        # --- Iterate critic update ---
        (rng, agent_state), critic_losses = jax.lax.scan(
            _update_critics,
            (rng, agent_state),
            None,
            length=args.num_critic_updates_per_step,
        )
        losses.update(jax.tree_map(jnp.mean, critic_losses))  # Average across updates

        # --- Update value function ---
        if args.use_awr or args.use_value_target:

            @jax.value_and_grad
            def _value_loss_fn(params):
                q_params = agent_state.vec_q.params
                targ = q_apply_fn(q_params, batch.obs, batch.action)
                adv = targ.min(-1) - value_apply_fn(params, batch.obs)
                # Asymmetric L2 loss
                expectile_weight = args.value_expectile - (adv < 0.0).astype(float)
                value_loss = jnp.abs(expectile_weight) * (adv**2)
                return jnp.mean(value_loss)

            value_loss, value_grad = _value_loss_fn(agent_state.value.params)
            agent_state = agent_state._replace(
                value=agent_state.value.apply_gradients(grads=value_grad),
            )
            losses.update({"value_loss": value_loss})

        # --- Update target networks ---
        def _update_target(state, target_state):
            new_params = optax.incremental_update(
                state.params, target_state.params, args.polyak_step_size
            )
            return target_state.replace(step=target_state.step + 1, params=new_params)

        new_q_target = _update_target(agent_state.vec_q, agent_state.vec_q_target)
        agent_state = agent_state._replace(vec_q_target=new_q_target)
        if args.use_target_actor:
            new_pi_target = _update_target(agent_state.actor, agent_state.actor_target)
            agent_state = agent_state._replace(actor_target=new_pi_target)

        return (rng, agent_state, rollout_buffer), jax.tree_map(jnp.mean, losses)

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

    # --- Initialize networks ---
    num_actions = env.single_action_space.shape[0]
    data_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), dataset)
    data_std = jax.tree_map(lambda x: jnp.std(x, axis=0), dataset)
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_net = Actor(
        args=args, data_mean=data_mean, data_std=data_std, num_actions=num_actions
    )
    rng, rng_act = jax.random.split(rng)
    actor = create_train_state(args, rng_act, actor_net, [dummy_obs], True)
    if args.use_target_actor:
        # Target networks share seeds to match initialization
        actor_target = create_train_state(args, rng_act, actor_net, [dummy_obs], True)
    else:
        actor_target = None
    q_net = VectorQ(args=args, data_mean=data_mean, data_std=data_std)
    if args.use_awr or args.use_value_target:
        value_net = StateValueFunction(
            args=args, data_mean=data_mean, data_std=data_std
        )
        rng, rng_value = jax.random.split(rng)
        value = create_train_state(args, rng_value, value_net, [dummy_obs])
        value_apply_fn = value_net.apply
    else:
        value, value_apply_fn = None, None
    if args.use_entropy_loss:
        alpha_net = EntropyCoef(args=args)
        rng, rng_alpha = jax.random.split(rng)
        alpha = create_train_state(args, rng_alpha, alpha_net, [])
        alpha_apply_fn = alpha_net.apply
    else:
        alpha, alpha_apply_fn = None, None
    rng, rng_q = jax.random.split(rng)
    # Increase steps for iterated critic updates
    nq = args.num_updates * args.num_critic_updates_per_step
    vec_q = create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action], steps=nq)
    agent_state = AgentTrainState(
        actor=actor,
        actor_target=actor_target,
        vec_q=vec_q,
        vec_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        value=value,
        alpha=alpha,
    )

    # --- Initialize buffer and rollout function ---
    if args.dataset_sample_ratio == 1.0:
        rollout_buffer, rollout_fn = None, None
    else:
        assert args.model_path, "Model path must be provided for model-based methods"
        dynamics_model = load_dynamics_model(args.model_path)
        dynamics_model.dataset = dataset
        max_buffer_size = args.rollout_batch_size * args.rollout_length
        max_buffer_size *= args.model_retain_epochs
        rollout_buffer = jax.tree_map(
            lambda x: jnp.zeros((max_buffer_size, *x.shape[1:])),
            dataset,
        )
        rollout_fn = dynamics_model.make_rollout_fn(
            batch_size=args.rollout_batch_size,
            rollout_length=args.rollout_length,
            step_penalty_coef=args.step_penalty_coef,
        )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args,
        actor_net.apply,
        q_net.apply,
        value_apply_fn,
        alpha_apply_fn,
        dataset,
        rollout_fn,
    )
    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state, rollout_buffer), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state, rollout_buffer),
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
        rets = onp.array([eval_agent(args, _rng, env, agent_state) for _rng in _rng])
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

    env.close()
    if args.log:
        wandb.finish()
