r""""Environment utilities for Project Phoenix.

Author:     Sven Gronauer
Created:    17.05.2021
"""
import os


def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'assets')
    return data_path
