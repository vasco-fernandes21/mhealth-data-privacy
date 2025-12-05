import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
import sys
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing.sleep_edf import load_windowed_sleep_edf
from preprocessing.wesad import load_processed_wesad

from ..core.config import settings


class DataLoader:
    _instance = None
    _cache: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        try:
            dataset_dir = settings.DATA_DIR / dataset_name.lower()
            
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            
            if dataset_name.lower() == 'sleep-edf':
                data_tuple = load_windowed_sleep_edf(str(dataset_dir))
                X_train, X_val, X_test, y_train, y_val, y_test, scaler, info, subjects_all = data_tuple
                subjects_train = np.load(dataset_dir / 'subjects_train.npy', allow_pickle=True)
                subjects_val = np.load(dataset_dir / 'subjects_val.npy', allow_pickle=True)
                subjects_test = np.load(dataset_dir / 'subjects_test.npy', allow_pickle=True)
            elif dataset_name.lower() == 'wesad':
                data_tuple = load_processed_wesad(str(dataset_dir))
                X_train, X_val, X_test, y_train, y_val, y_test, scaler, info = data_tuple
                subjects_train = np.load(dataset_dir / 'subjects_train.npy', allow_pickle=True)
                subjects_val = np.load(dataset_dir / 'subjects_val.npy', allow_pickle=True)
                subjects_test = np.load(dataset_dir / 'subjects_test.npy', allow_pickle=True)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_train = y_train.astype(np.int64)
            y_val = y_val.astype(np.int64)
            y_test = y_test.astype(np.int64)
            
            def convert_subjects_to_ids(subjects_arr):
                if subjects_arr.dtype == object or subjects_arr.dtype.kind in ['U', 'S']:
                    unique_subs = np.unique(subjects_arr)
                    sub_to_id = {sub: i for i, sub in enumerate(unique_subs)}
                    return np.array([sub_to_id[sub] for sub in subjects_arr], dtype=np.int64)
                else:
                    return subjects_arr.astype(np.int64)
            
            subjects_train = convert_subjects_to_ids(subjects_train)
            subjects_val = convert_subjects_to_ids(subjects_val)
            subjects_test = convert_subjects_to_ids(subjects_test)
            
            X_train_full = np.concatenate([X_train, X_val], axis=0)
            y_train_full = np.concatenate([y_train, y_val], axis=0)
            subjects_train_full = np.concatenate([subjects_train, subjects_val], axis=0)
            
            input_dim = X_train.shape[1]
            n_classes = len(np.unique(y_train_full))
            
            dataset = {
                "train": (X_train, y_train, subjects_train),
                "train_full": (X_train_full, y_train_full, subjects_train_full),
                "val": (X_val, y_val, subjects_val),
                "test": (X_test, y_test),
                "input_dim": input_dim,
                "n_classes": n_classes
            }
            
            self._cache[dataset_name] = dataset
            return dataset

        except FileNotFoundError as e:
            raise RuntimeError(f"Data not found in {settings.DATA_DIR}. {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {dataset_name}: {str(e)}")


data_loader = DataLoader()

