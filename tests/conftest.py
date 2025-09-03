import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for tests."""
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()
    return temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "algorithm": "sac",
        "env_name": "HalfCheetah-v2",
        "seed": 42,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "hidden_dims": [256, 256],
        "max_steps": 1000000,
        "eval_frequency": 5000,
        "num_eval_episodes": 10,
    }


@pytest.fixture
def sample_observation() -> np.ndarray:
    """Provide a sample observation array for RL testing."""
    return np.random.randn(17)  # HalfCheetah observation size


@pytest.fixture
def sample_action() -> np.ndarray:
    """Provide a sample action array for RL testing."""
    return np.random.randn(6)  # HalfCheetah action size


@pytest.fixture
def sample_batch_data() -> Dict[str, np.ndarray]:
    """Provide a sample batch of RL data."""
    batch_size = 32
    obs_dim = 17
    action_dim = 6
    
    return {
        "observations": np.random.randn(batch_size, obs_dim),
        "actions": np.random.randn(batch_size, action_dim),
        "rewards": np.random.randn(batch_size),
        "next_observations": np.random.randn(batch_size, obs_dim),
        "dones": np.random.choice([True, False], size=batch_size),
    }


@pytest.fixture
def mock_env_config():
    """Mock environment configuration."""
    return {
        "env_name": "HalfCheetah-v2",
        "max_episode_steps": 1000,
        "normalize_obs": True,
        "normalize_reward": True,
    }


@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    return {
        "hidden_dims": [256, 256],
        "activation": "relu",
        "layer_norm": True,
        "dropout_rate": 0.0,
    }


@pytest.fixture
def random_seed():
    """Set a random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture(autouse=True)
def clean_env_vars():
    """Clean environment variables before each test."""
    env_vars_to_clean = [
        "WANDB_MODE",
        "JAX_PLATFORM_NAME",
        "CUDA_VISIBLE_DEVICES",
    ]
    
    original_values = {}
    for var in env_vars_to_clean:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    # Set test-friendly defaults
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    yield
    
    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_dataset_path(temp_dir: Path) -> Path:
    """Create a mock dataset file path."""
    dataset_path = temp_dir / "mock_dataset.pkl"
    return dataset_path


@pytest.fixture
def small_batch_size():
    """Provide a small batch size for quick tests."""
    return 8


@pytest.fixture
def medium_batch_size():
    """Provide a medium batch size for integration tests."""
    return 32


@pytest.fixture
def large_batch_size():
    """Provide a large batch size for performance tests."""
    return 256


class MockEnvironment:
    """Mock environment for testing RL algorithms."""
    
    def __init__(self, obs_dim: int = 17, action_dim: int = 6):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        """Reset the environment."""
        self._state = np.random.randn(self.obs_dim)
        return self._state.copy()
    
    def step(self, action: np.ndarray):
        """Take a step in the environment."""
        next_state = np.random.randn(self.obs_dim)
        reward = np.random.randn()
        done = np.random.choice([True, False], p=[0.05, 0.95])  # 5% chance of episode end
        info = {}
        
        self._state = next_state
        return next_state.copy(), reward, done, info
    
    @property
    def observation_space(self):
        """Mock observation space."""
        class MockSpace:
            shape = (self.obs_dim,)
            dtype = np.float32
        return MockSpace()
    
    @property
    def action_space(self):
        """Mock action space."""
        class MockSpace:
            shape = (self.action_dim,)
            dtype = np.float32
        return MockSpace()


@pytest.fixture
def mock_env():
    """Provide a mock environment for testing."""
    return MockEnvironment()


@pytest.fixture
def trajectory_data():
    """Generate sample trajectory data."""
    trajectory_length = 100
    obs_dim = 17
    action_dim = 6
    
    observations = np.random.randn(trajectory_length, obs_dim)
    actions = np.random.randn(trajectory_length, action_dim)
    rewards = np.random.randn(trajectory_length)
    dones = np.zeros(trajectory_length, dtype=bool)
    dones[-1] = True  # End the trajectory
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "next_observations": np.roll(observations, -1, axis=0),
    }


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent