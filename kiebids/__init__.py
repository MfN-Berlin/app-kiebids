import os
from importlib import metadata
from importlib.metadata import version

import yaml
from dotenv import load_dotenv
from dotmap import DotMap

from kiebids.logger import getLogger

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "../configs/default_config.yml")) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), "../configs/ocr_config.yaml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

__all__ = [
    "getLogger",
]
