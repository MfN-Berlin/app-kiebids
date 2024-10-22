import os

import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger
import fiftyone as fo
import fiftyone.core.dataset as fod

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "../configs/default_config.yml")) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), "../configs/ocr_config.yaml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

current_dataset = fod.Dataset(name="mfn-dataset", overwrite=True)

__all__ = [
    "get_logger",
]
