"""
Validation tests to ensure the testing infrastructure is properly set up.
"""
import pytest
import numpy as np
from pathlib import Path


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


def test_pytest_markers():
    """Test that custom pytest markers are working."""
    pass


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker is working."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker is working."""
    assert "hello" + " " + "world" == "hello world"


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker is working."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True


def test_fixtures_available(temp_dir, sample_config, sample_observation):
    """Test that shared fixtures are available and working."""
    # Test temp_dir fixture
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test sample_config fixture
    assert isinstance(sample_config, dict)
    assert "algorithm" in sample_config
    assert "env_name" in sample_config
    
    # Test sample_observation fixture
    assert isinstance(sample_observation, np.ndarray)
    assert len(sample_observation) > 0


def test_mock_env_fixture(mock_env):
    """Test that mock environment fixture works."""
    # Test environment reset
    obs = mock_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (17,)  # HalfCheetah obs dim
    
    # Test environment step
    action = np.random.randn(6)  # HalfCheetah action dim
    next_obs, reward, done, info = mock_env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (17,)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_batch_data_fixture(sample_batch_data):
    """Test that batch data fixture provides correct format."""
    required_keys = ["observations", "actions", "rewards", "next_observations", "dones"]
    
    for key in required_keys:
        assert key in sample_batch_data
        assert isinstance(sample_batch_data[key], np.ndarray)
    
    # Check shapes are consistent
    batch_size = len(sample_batch_data["observations"])
    assert len(sample_batch_data["actions"]) == batch_size
    assert len(sample_batch_data["rewards"]) == batch_size
    assert len(sample_batch_data["next_observations"]) == batch_size
    assert len(sample_batch_data["dones"]) == batch_size


def test_trajectory_data_fixture(trajectory_data):
    """Test that trajectory data fixture provides correct format."""
    required_keys = ["observations", "actions", "rewards", "dones", "next_observations"]
    
    for key in required_keys:
        assert key in trajectory_data
        assert isinstance(trajectory_data[key], np.ndarray)
    
    # Check that trajectory ends with done=True
    assert trajectory_data["dones"][-1] == True


def test_config_fixtures(mock_env_config, mock_model_config):
    """Test that configuration fixtures are properly structured."""
    # Test environment config
    assert "env_name" in mock_env_config
    assert "max_episode_steps" in mock_env_config
    
    # Test model config  
    assert "hidden_dims" in mock_model_config
    assert "activation" in mock_model_config


def test_random_seed_fixture(random_seed):
    """Test that random seed fixture sets reproducible randomness."""
    assert isinstance(random_seed, int)
    
    # Generate some random numbers and verify they're reproducible
    np.random.seed(random_seed)
    random_nums1 = np.random.randn(5)
    
    np.random.seed(random_seed)
    random_nums2 = np.random.randn(5)
    
    np.testing.assert_array_equal(random_nums1, random_nums2)


def test_batch_size_fixtures(small_batch_size, medium_batch_size, large_batch_size):
    """Test that batch size fixtures provide reasonable values."""
    assert small_batch_size < medium_batch_size < large_batch_size
    assert small_batch_size > 0
    assert isinstance(small_batch_size, int)
    assert isinstance(medium_batch_size, int)
    assert isinstance(large_batch_size, int)


def test_project_structure(project_root):
    """Test that project structure is as expected."""
    assert project_root.exists()
    assert project_root.is_dir()
    
    # Check for key project files/directories
    assert (project_root / "algorithms").exists()
    assert (project_root / "configs").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "pyproject.toml").exists()


def test_numpy_available():
    """Test that numpy is properly installed and importable."""
    import numpy as np
    assert hasattr(np, "__version__")
    
    # Test basic numpy operations
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    assert arr.mean() == 3.0


def test_pytest_mock_available():
    """Test that pytest-mock is available."""
    pytest_mock = pytest.importorskip("pytest_mock")
    assert hasattr(pytest_mock, "MockerFixture")


class TestInfrastructureValidation:
    """Test class to validate testing infrastructure setup."""
    
    def test_class_structure(self):
        """Test that test classes work properly."""
        assert True
    
    def test_method_discovery(self):
        """Test that test methods are discovered correctly."""
        assert hasattr(self, "test_class_structure")
    
    @pytest.mark.unit
    def test_class_with_markers(self):
        """Test that markers work with test classes."""
        assert True


def test_coverage_includes_algorithms():
    """Test that our coverage configuration will include the algorithms package."""
    # This is more of a structural test - we're not actually measuring coverage here
    # but ensuring our test can import from the algorithms package
    from pathlib import Path
    algorithms_dir = Path(__file__).parent.parent / "algorithms"
    assert algorithms_dir.exists()
    assert algorithms_dir.is_dir()
    
    # Check that there are Python files in algorithms
    python_files = list(algorithms_dir.glob("*.py"))
    assert len(python_files) > 0