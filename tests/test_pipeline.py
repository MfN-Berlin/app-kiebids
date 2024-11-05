import pytest

from kiebids import config, get_logger

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def test_basic():
    pass


if __name__ == "__main__":
    pytest.main(["-s", "test_pipeline.py"])
