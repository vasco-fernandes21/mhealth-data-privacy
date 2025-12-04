import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from ..core.config import settings


class DataLoader:
    _instance = None
    _cache: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load dataset from processed directory structure."""
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        try:
            # Data is organized in subdirectories: data/processed/wesad/ or data/processed/sleep-edf/
            dataset_dir = settings.DATA_DIR / dataset_name.lower()
            
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            
            # Load train/val/test splits
            X_train = np.load(dataset_dir / 'X_train.npy', allow_pickle=False).astype(np.float32)
            y_train = np.load(dataset_dir / 'y_train.npy', allow_pickle=False).astype(np.int64)
            # Subjects arrays contain object types (strings), need allow_pickle=True
            subjects_train = np.load(dataset_dir / 'subjects_train.npy', allow_pickle=True)
            
            X_val = np.load(dataset_dir / 'X_val.npy', allow_pickle=False).astype(np.float32)
            y_val = np.load(dataset_dir / 'y_val.npy', allow_pickle=False).astype(np.int64)
            subjects_val = np.load(dataset_dir / 'subjects_val.npy', allow_pickle=True)
            
            X_test = np.load(dataset_dir / 'X_test.npy', allow_pickle=False).astype(np.float32)
            y_test = np.load(dataset_dir / 'y_test.npy', allow_pickle=False).astype(np.int64)
            subjects_test = np.load(dataset_dir / 'subjects_test.npy', allow_pickle=True)
            
            # Convert subjects to numeric IDs if they are strings/objects
            # This ensures we can use them for splitting in FL
            def convert_subjects_to_ids(subjects_arr):
                # Handle both string arrays and numeric arrays
                if subjects_arr.dtype == object or subjects_arr.dtype.kind in ['U', 'S']:
                    # String/object array - convert to numeric IDs
                    unique_subs = np.unique(subjects_arr)
                    sub_to_id = {sub: i for i, sub in enumerate(unique_subs)}
                    return np.array([sub_to_id[sub] for sub in subjects_arr], dtype=np.int64)
                else:
                    # Already numeric, just ensure int64
                    return subjects_arr.astype(np.int64)
            
            subjects_train = convert_subjects_to_ids(subjects_train)
            subjects_val = convert_subjects_to_ids(subjects_val)
            subjects_test = convert_subjects_to_ids(subjects_test)
            
            # Combine train and val for FL training (test is kept separate)
            X_train_full = np.concatenate([X_train, X_val], axis=0)
            y_train_full = np.concatenate([y_train, y_val], axis=0)
            subjects_train_full = np.concatenate([subjects_train, subjects_val], axis=0)
            
            # Get metadata
            input_dim = X_train.shape[1]
            n_classes = len(np.unique(y_train_full))
            
            dataset = {
                "train": (X_train_full, y_train_full, subjects_train_full),
                "test": (X_test, y_test),
                "input_dim": input_dim,
                "n_classes": n_classes
            }
            
            self._cache[dataset_name] = dataset
            print(f"Loaded {dataset_name}: {X_train_full.shape[0]} train samples, {X_test.shape[0]} test samples")
            print(f"   Input dim: {input_dim}, Classes: {n_classes}")
            return dataset

        except FileNotFoundError as e:
            raise RuntimeError(f"Data not found in {settings.DATA_DIR}. {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {dataset_name}: {str(e)}")


data_loader = DataLoader()

