"""Evaluation utilities for the Unifloral project.

This module provides tools for:
1. Loading and parsing experiment results
2. Running bandit-based policy selection
3. Computing confidence intervals via bootstrapping
"""

from collections import namedtuple
from datetime import datetime
import os
import re
from typing import Dict, Tuple
import warnings

from functools import partial
import glob
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Data loading
"""


def parse_and_load_npz(filename: str) -> Dict:
    """Load data from a result file and parse metadata from filename.

    Args:
        filename: Path to the .npz result file

    Returns:
        Dictionary containing loaded arrays and metadata
    """
    # Parse filename to extract algorithm, dataset, and timestamp
    pattern = r"(.+)_(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.match(pattern, os.path.basename(filename))
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")

    algorithm, dataset, dt_str = match.groups()
    dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")

    data = np.load(filename, allow_pickle=True)
    data = {k: v for k, v in data.items()}
    data["algorithm"] = algorithm
    data["dataset"] = dataset
    data["datetime"] = dt
    data.update(data.pop("args", np.array({})).item())  # Flatten args
    return data


def load_results_dataframe(results_dir: str = "final_returns") -> pd.DataFrame:
    """Load all result files from a directory into a pandas DataFrame.

    Args:
        results_dir: Directory containing .npz result files

    Returns:
        DataFrame containing results from all successfully loaded files
    """
    npz_files = glob.glob(os.path.join(results_dir, "*.npz"))
    data_list = []

    for f in npz_files:
        try:
            data = parse_and_load_npz(f)
            data_list.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    df = pd.DataFrame(data_list).drop(columns=["Index"], errors="ignore")
    if "final_scores" in df.columns:
        df["final_scores"] = df["final_scores"].apply(lambda x: x.reshape(-1))
    if "final_returns" in df.columns:
        df["final_returns"] = df["final_returns"].apply(lambda x: x.reshape(-1))

    df = df.sort_values(by=["algorithm", "dataset", "datetime"])
    return df.reset_index(drop=True)


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
      \___/     Bandit Evaluation and Bootstrapping
"""

BanditState = namedtuple("BanditState", "rng counts rewards total_pulls")


def ucb(
    means: jnp.ndarray, counts: jnp.ndarray, total_counts: int, alpha: float
) -> jnp.ndarray:
    """Compute UCB exploration bonus.

    Args:
        means: Array of empirical means for each arm
        counts: Array of pull counts for each arm
        total_counts: Total number of pulls across all arms
        alpha: Exploration coefficient

    Returns:
        Array of UCB values for each arm
    """
    exploration = jnp.sqrt(alpha * jnp.log(total_counts) / (counts + 1e-9))
    return means + exploration


def argmax_with_random_tiebreaking(rng: jnp.ndarray, values: jnp.ndarray) -> int:
    """Select maximum value with random tiebreaking.

    Args:
        rng: JAX PRNGKey
        values: Array of values to select from

    Returns:
        Index of selected maximum value
    """
    mask = values == jnp.max(values)
    p = mask / (mask.sum() + 1e-9)
    return jax.random.choice(rng, jnp.arange(len(values)), p=p)


@partial(jax.jit, static_argnums=(2,))
def run_bandit(
    returns_array: jnp.ndarray,
    rng: jnp.ndarray,
    max_pulls: int,
    alpha: float,
    policy_idx: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single bandit algorithm and report results after each pull.

    Args:
        returns_array: Array of returns for each policy and rollout
        rng: JAX PRNGKey
        max_pulls: Maximum number of pulls to execute
        alpha: UCB exploration coefficient
        policy_idx: Indices of policies to consider

    Returns:
        Tuple of (pulls, estimated_bests)
    """
    returns_array = returns_array[policy_idx]
    num_policies, num_rollouts = returns_array.shape

    init_state = BanditState(
        rng=rng,
        counts=jnp.zeros(num_policies, dtype=jnp.int32),
        rewards=jnp.zeros(num_policies),
        total_pulls=1,
    )

    def bandit_step(state: BanditState, _):
        """Run one bandit step and track performance."""
        rng, rng_lever, rng_reward = jax.random.split(state.rng, 3)

        # Select arm using UCB
        means = state.rewards / jnp.maximum(state.counts, 1)
        ucb_values = ucb(means, state.counts, state.total_pulls, alpha)
        arm = argmax_with_random_tiebreaking(rng_lever, ucb_values)

        # Sample a reward for the chosen arm
        idx = jax.random.randint(rng_reward, shape=(), minval=0, maxval=num_rollouts)
        reward = returns_array[arm, idx]
        new_state = BanditState(
            rng=rng,
            counts=state.counts.at[arm].add(1),
            rewards=state.rewards.at[arm].add(reward),
            total_pulls=state.total_pulls + 1,
        )

        # Calculate best arm based on current state
        updated_means = new_state.rewards / jnp.maximum(new_state.counts, 1)
        best_arm = jnp.argmax(updated_means)
        estimated_best = returns_array[best_arm].mean()

        return new_state, (state.total_pulls, estimated_best)

    _, (pulls, estimated_bests) = jax.lax.scan(
        bandit_step, init_state, length=max_pulls
    )
    return pulls, estimated_bests


def run_bandit_trials(
    returns_array: jnp.ndarray,
    seed: int = 17,
    num_subsample: int = 20,
    num_repeats: int = 1000,
    max_pulls: int = 200,
    ucb_alpha: float = 2.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run multiple bandit trials and collect results at each step.

    Args:
        returns_array: Array of returns for each policy and rollout
        seed: Random seed
        num_subsample: Number of policies to subsample on each trial
        num_repeats: Number of trials to run
        max_pulls: Maximum number of pulls per trial
        ucb_alpha: UCB exploration coefficient

    Returns:
        Tuple of (pulls, estimated_bests)
    """
    rng = jax.random.PRNGKey(seed)
    num_policies = returns_array.shape[0]

    num_subsample = min(num_subsample, num_policies)
    if num_subsample > num_policies:
        warnings.warn("Not enough policies to subsample, using all policies")

    rng, rng_trials, rng_sample = jax.random.split(rng, 3)
    rng_trials = jax.random.split(rng_trials, num_repeats)

    def sample_policies(rng: jnp.ndarray) -> jnp.ndarray:
        """Sample a subset of policy indices."""
        if num_subsample > num_policies:
            return jnp.arange(num_policies)
        return jax.random.choice(
            rng, jnp.arange(num_policies), shape=(num_subsample,), replace=False
        )

    # Create a batch of policy index arrays for all trials
    rng_sample_keys = jax.random.split(rng_sample, num_repeats)
    policy_indices = jax.vmap(sample_policies)(rng_sample_keys)

    # Run bandit trials with policy subsampling
    # Pulls are the same for all trials, so we can just return the first one
    vmap_run_bandit = jax.vmap(run_bandit, in_axes=(None, 0, None, None, 0))
    pulls, estimated_bests = vmap_run_bandit(
        returns_array, rng_trials, max_pulls, ucb_alpha, policy_indices
    )
    return pulls[0], estimated_bests


def bootstrap_confidence_interval(
    rng: jnp.ndarray,
    data: jnp.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean of data.

    Args:
        rng: JAX PRNGKey
        data: Array of values to bootstrap
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level (between 0 and 1)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """

    @jax.vmap
    def bootstrap_mean(rng):
        samples = jax.random.choice(rng, data, shape=(data.shape[0],), replace=True)
        return samples.mean()

    bootstrap_means = bootstrap_mean(jax.random.split(rng, n_bootstraps))
    lower_bound = jnp.percentile(bootstrap_means, 100 * (1 - confidence) / 2)
    upper_bound = jnp.percentile(bootstrap_means, 100 * (1 + confidence) / 2)
    return lower_bound, upper_bound


def bootstrap_bandit_trials(
    returns_array: jnp.ndarray,
    seed: int = 17,
    num_subsample: int = 20,
    num_repeats: int = 1000,
    max_pulls: int = 200,
    ucb_alpha: float = 2.0,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Run bandit trials and compute bootstrap confidence intervals.

    Args:
        returns_array: Array of returns for each policy and rollout has shape (num_policies, num_rollouts)
        seed: Random seed
        num_subsample: Number of policies to subsample
        num_repeats: Number of bandit trials to run
        max_pulls: Maximum number of pulls per trial
        ucb_alpha: UCB exploration coefficient
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level for intervals

    Returns:
        Dictionary with the following keys:
        - pulls: Number of pulls at each step
        - estimated_bests_mean: Mean of the currently estimated best returns across trials
        - estimated_bests_ci_low: Lower confidence bound for estimated best returns
        - estimated_bests_ci_high: Upper confidence bound for estimated best returns
    """
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, max_pulls)

    pulls, estimated_bests = run_bandit_trials(
        returns_array, seed, num_subsample, num_repeats, max_pulls, ucb_alpha
    )
    vmap_bootstrap = jax.vmap(bootstrap_confidence_interval, in_axes=(0, 1, None, None))
    ci_low, ci_high = vmap_bootstrap(rng, estimated_bests, n_bootstraps, confidence)
    estimated_bests_mean = estimated_bests.mean(axis=0)

    return {
        "pulls": pulls,
        "estimated_bests_mean": estimated_bests_mean,
        "estimated_bests_ci_low": ci_low,
        "estimated_bests_ci_high": ci_high,
    }
