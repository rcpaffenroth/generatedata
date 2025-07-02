"""
Unit tests for generatedata.config
"""
from generatedata import config

def test_load_config():
    # Assert that config.py defines the symbol DATA_URL
    assert hasattr(config, 'DATA_URL')
    assert isinstance(config.DATA_URL, str)
