#!/usr/bin/env python3
"""Setup configuration for MHealth Privacy package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text() if readme_path.exists() else 'Privacy-Preserving Mobile Health Data Analysis'

# Read requirements if exists
requirements = []
requirements_path = Path(__file__).parent / 'requirements.txt'
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    # Remove empty lines and comments
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name='mhealth-privacy',
    version='0.1.0',
    description='Privacy-Preserving Analysis for Mobile Health Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/mhealth-privacy',
    license='MIT',
    
    # Find all packages under src/
    packages=find_packages(),
    
    python_requires='>=3.8',
    install_requires=requirements,
    
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=3.0',
        ],
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)