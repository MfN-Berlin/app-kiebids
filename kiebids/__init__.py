import os

import fiftyone.core.dataset as fod
import yaml
from dotenv import load_dotenv
from dotmap import DotMap
from prefect.logging import get_logger

load_dotenv()

ocr_config = os.getenv("OCR_CONFIG", "ocr_config.yaml")
workflow_config = os.getenv("WORKFLOW_CONFIG", "workflow_config.yaml")
with open(
    os.path.join(os.path.dirname(__file__), f"../configs/{workflow_config}")
) as f:
    config = DotMap(yaml.safe_load(f))

with open(os.path.join(os.path.dirname(__file__), f"../configs/{ocr_config}")) as f:
    pipeline_config = DotMap(yaml.safe_load(f))

current_dataset = None
if not config.disable_fiftyone:
    os.environ["FIFTYONE_DATABASE_DIR"] = config.fiftyone_database_dir

    current_dataset = fod.load_dataset(
        config.fiftyone_dataset, create_if_necessary=True
    )
    current_dataset.overwrite = True
    current_dataset.persistent = True

__all__ = [
    "get_logger",
]
