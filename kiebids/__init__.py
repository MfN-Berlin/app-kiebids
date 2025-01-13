import os
from datetime import datetime

import fiftyone.core.dataset as fod
import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

load_dotenv()

ocr_config = os.getenv("OCR_CONFIG", "ocr_config.yaml")
workflow_config = os.getenv("WORKFLOW_CONFIG", "workflow_config.yaml")
with open(
    os.path.join(os.path.dirname(__file__), f"../configs/{workflow_config}")
) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), f"../configs/{ocr_config}")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

fiftyone_dataset = None
if not config.disable_fiftyone:
    os.environ["FIFTYONE_DATABASE_DIR"] = config.fiftyone_database_dir

    fiftyone_dataset = fod.load_dataset(
        config.fiftyone_dataset, create_if_necessary=True
    )
    fiftyone_dataset.overwrite = True
    fiftyone_dataset.persistent = True

if config.evaluation:
    log_dir = f"{config.evaluation_path}/tensorboard/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    evaluation_writer = SummaryWriter(log_dir)
    event_accumulator = EventAccumulator(log_dir)
else:
    evaluation_writer = None

__all__ = [
    "get_logger",
]
