import os

import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger
import fiftyone.core.dataset as fod

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "../configs/default_config.yml")) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), "../configs/ocr_config.yaml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

if fod.dataset_exists(config.fiftyone_dataset):
    current_dataset = fod.load_dataset(config.fiftyone_dataset)
else:
    current_dataset = fod.Dataset(name=config.fiftyone_dataset, overwrite=True, persistent=True)

__all__ = [
    "get_logger",
]
