"""
Setup configuration for mhealth-data-privacy project.
Allows installation as a package for easy imports in notebooks.
"""

from setuptools import setup, find_packages

setup(
    name="mhealth_privacy",
    version="0.1.0",
    description="Privacy-Preserving Health Data Analysis with DP and FL",
    author="Eduardo Carvalho, Filipe Correia, Vasco Fernandes",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "tensorflow-privacy>=0.8.0",
        "flwr[simulation]>=1.0.0",
        "mne>=1.0.0",
        "pyedflib>=0.1.30",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

