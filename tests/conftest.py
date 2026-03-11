"""Pytest configuration — add tests/ to sys.path for helpers/data imports."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
