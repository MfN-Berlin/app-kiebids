import os
from tensorboardX import SummaryWriter
from datetime import datetime

import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger

load_dotenv()

with open(os.path.join(os.path.dirname(__file__), "../configs/default_config.yml")) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), "../configs/ocr_config.yaml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

log_dir = f"{config.evaluation}/tensorboard/layout_analysis_" + datetime.now().strftime("%Y%m%d-%H%M%S")
evaluation_writer = SummaryWriter(log_dir)

__all__ = [
    "get_logger",
]
