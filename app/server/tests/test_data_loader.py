import pytest
import numpy as np
from unittest.mock import patch
from pathlib import Path
from src.data.loader import DataLoader


class TestDataLoader:
    def test_singleton_pattern(self):
        loader1 = DataLoader()
        loader2 = DataLoader()
        assert loader1 is loader2
    
    @patch('src.data.loader.load_windowed_sleep_edf')
    @patch('src.data.loader.np.load')
    def test_load_sleep_edf_dataset(self, mock_np_load, mock_load_edf):
        mock_load_edf.return_value = (
            np.random.randn(100, 24).astype(np.float32),
            np.random.randn(20, 24).astype(np.float32),
            np.random.randn(30, 24).astype(np.float32),
            np.random.randint(0, 5, 100).astype(np.int64),
            np.random.randint(0, 5, 20).astype(np.int64),
            np.random.randint(0, 5, 30).astype(np.int64),
            None,
            {"n_classes": 5},
            np.zeros(100)
        )
        mock_np_load.side_effect = [
            np.zeros(100),
            np.zeros(20),
            np.zeros(30)
        ]
        
        loader = DataLoader()
        loader._cache = {}
        
        with patch('src.data.loader.settings.DATA_DIR', Path('/fake/path')):
            with patch('pathlib.Path.exists', return_value=True):
                dataset = loader.load_dataset("sleep-edf")
        
        assert "train" in dataset
        assert "test" in dataset
        assert dataset["input_dim"] == 24
        assert dataset["n_classes"] == 5
    
    @patch('src.data.loader.load_processed_wesad')
    @patch('src.data.loader.np.load')
    def test_load_wesad_dataset(self, mock_np_load, mock_load_wesad):
        mock_load_wesad.return_value = (
            np.random.randn(100, 140).astype(np.float32),
            np.random.randn(20, 140).astype(np.float32),
            np.random.randn(30, 140).astype(np.float32),
            np.random.randint(0, 2, 100).astype(np.int64),
            np.random.randint(0, 2, 20).astype(np.int64),
            np.random.randint(0, 2, 30).astype(np.int64),
            None,
            {"n_classes": 2}
        )
        mock_np_load.side_effect = [
            np.zeros(100),
            np.zeros(20),
            np.zeros(30)
        ]
        
        loader = DataLoader()
        loader._cache = {}
        
        with patch('src.data.loader.settings.DATA_DIR', Path('/fake/path')):
            with patch('pathlib.Path.exists', return_value=True):
                dataset = loader.load_dataset("wesad")
        
        assert "train" in dataset
        assert dataset["input_dim"] == 140
        assert dataset["n_classes"] == 2
    
    def test_cache_functionality(self):
        loader = DataLoader()
        loader._cache = {}
        
        fake_dataset = {
            "train": (np.array([1, 2, 3]), np.array([0, 1, 0]), np.array([0, 0, 0])),
            "test": (np.array([4, 5]), np.array([1, 0])),
            "input_dim": 1,
            "n_classes": 2
        }
        
        loader._cache["test_dataset"] = fake_dataset
        result = loader.load_dataset("test_dataset")
        assert result == fake_dataset
    
    def test_unknown_dataset_raises_error(self):
        loader = DataLoader()
        loader._cache = {}
        
        with patch('src.data.loader.settings.DATA_DIR', Path('/fake/path')):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(RuntimeError, match="Unknown dataset"):
                    loader.load_dataset("unknown_dataset")

