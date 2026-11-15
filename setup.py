"""Legacy setup.py for editable installs and tools that don't yet read pyproject.toml.

Prefer: pip install -e .   (uses pyproject.toml)
Fallback: python setup.py develop
"""
from setuptools import setup

setup()
