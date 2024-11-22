import os
from datetime import datetime

import fiftyone.core.dataset as fod
import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger
from tensorboardX import SummaryWriter

load_dotenv()

with open(
    os.path.join(os.path.dirname(__file__), "../configs/workflow_config.yaml")
) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), "../configs/ocr_config.yaml")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

os.environ["FIFTYONE_DATABASE_DIR"] = config.fiftyone_database_dir

fiftyone_dataset = fod.load_dataset(config.fiftyone_dataset, create_if_necessary=True)
fiftyone_dataset.overwrite = True
fiftyone_dataset.persistent = True

# log_dir = (
#     f"{config.evaluation_path}/tensorboard/layout_analysis_"
#     + datetime.now().strftime("%Y%m%d-%H%M%S")
# )

log_dir = (
    f"{config.evaluation_path}/tensorboard/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
evaluation_writer = SummaryWriter(log_dir)

__all__ = [
    "get_logger",
]
