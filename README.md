<h1 align="center">🌹 Unifloral: Unified Offline Reinforcement Learning</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2504.11453">
        <img src="https://img.shields.io/badge/arXiv-2504.11453-b31b1b.svg" /></a>
</p>

Unified implementations and rigorous evaluation for offline reinforcement learning - built by [Matthew Jackson](https://github.com/EmptyJackson), [Uljad Berdica](https://github.com/uljad), and [Jarek Liesen](https://github.com/keraJLi).

## 💡 Code Philosophy

- ⚛️ **Single-file**: We implement algorithms as standalone Python files.
- 🤏 **Minimal**: We only edit what is necessary between algorithms, making comparisons straightforward.
- ⚡️ **GPU-accelerated**: We use JAX and end-to-end compile all training code, enabling lightning-fast training.

Inspired by [CORL](https://github.com/tinkoff-ai/CORL) and [CleanRL](https://github.com/vwxyzjn/cleanrl) - check them out!

## 🤖 Algorithms

We provide two types of algorithm implementation:

1. **Standalone**: Each algorithm is implemented as a [single file](algorithms) with minimal dependencies, making it easy to understand and modify.
2. **Unified**: Most algorithms are available as configs for our unified implementation [`unifloral.py`](algorithms/unifloral.py).

After training, final evaluation results are saved to `.npz` files in [`final_returns/`](final_returns) for analysis using our evaluation protocol.

All scripts support [D4RL](https://github.com/Farama-Foundation/D4RL) and use [Weights & Biases](https://wandb.ai) for logging, with configs provided as WandB sweep files.

### Model-free

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| BC | [`bc.py`](algorithms/bc.py) | [`unifloral/bc.yaml`](configs/unifloral/bc.yaml) | - |
| SAC-N | [`sac_n.py`](algorithms/sac_n.py) | [`unifloral/sac_n.yaml`](configs/unifloral/sac_n.yaml) | [[ArXiv]](https://arxiv.org/pdf/2110.01548) |
| EDAC | [`edac.py`](algorithms/edac.py) | [`unifloral/edac.yaml`](configs/unifloral/edac.yaml) | [[ArXiv]](https://arxiv.org/pdf/2110.01548) |
| CQL | [`cql.py`](algorithms/cql.py) | - | [[ArXiv]](https://arxiv.org/abs/2006.04779) |
| IQL | [`iql.py`](algorithms/iql.py) | [`unifloral/iql.yaml`](configs/unifloral/iql.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.06169) |
| TD3-BC | [`td3_bc.py`](algorithms/td3_bc.py) | [`unifloral/td3_bc.yaml`](configs/unifloral/td3_bc.yaml) | [[ArXiv]](https://arxiv.org/abs/2106.06860) |
| ReBRAC | [`rebrac.py`](algorithms/rebrac.py) | [`unifloral/rebrac.yaml`](configs/unifloral/rebrac.yaml) | [[ArXiv]](https://arxiv.org/abs/2305.09836) |
| TD3-AWR | - | [`unifloral/td3_awr.yaml`](configs/unifloral/td3_awr.yaml) | - |

### Model-based

We implement a single script for dynamics model training: [`dynamics.py`](algorithms/dynamics.py), with config [`dynamics.yaml`](configs/dynamics.yaml).

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| MOPO | [`mopo.py`](algorithms/mopo.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.13239) |
| MOReL | [`morel.py`](algorithms/morel.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.05951) |
| COMBO | [`combo.py`](algorithms/combo.py) | - | [[ArXiv]](https://arxiv.org/abs/2102.08363) |
| MoBRAC | - | [`unifloral/mobrac.yaml`](configs/unifloral/mobrac.yaml) | - |

New ones coming soon 👀

## 📊 Evaluation

Our evaluation script ([`evaluation.py`](evaluation.py)) implements the protocol described in our paper, analysing the performance of a UCB bandit over a range of policy evaluations.

```python
from evaluation import load_results_dataframe, bootstrap_bandit_trials
import jax.numpy as jnp

# Load all results from the final_returns directory
df = load_results_dataframe("final_returns")

# Run bandit trials with bootstrapped confidence intervals
results = bootstrap_bandit_trials(
    returns_array=jnp.array(policy_returns),  # Shape: (num_policies, num_rollouts)
    num_subsample=8,     # Number of policies to subsample
    num_repeats=1000,    # Number of bandit trials
    max_pulls=200,       # Maximum pulls per trial
    ucb_alpha=2.0,       # UCB exploration coefficient
    n_bootstraps=1000,   # Bootstrap samples for confidence intervals
    confidence=0.95      # Confidence level
)

# Access results
pulls = results["pulls"]                      # Number of pulls at each step
means = results["estimated_bests_mean"]       # Mean score of estimated best policy
ci_low = results["estimated_bests_ci_low"]    # Lower confidence bound
ci_high = results["estimated_bests_ci_high"]  # Upper confidence bound
```

## 📝 Cite us!
```bibtex
@misc{jackson2025clean,
      title={A Clean Slate for Offline Reinforcement Learning},
      author={Matthew Thomas Jackson and Uljad Berdica and Jarek Liesen and Shimon Whiteson and Jakob Nicolaus Foerster},
      year={2025},
      eprint={2504.11453},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.11453},
}
```
