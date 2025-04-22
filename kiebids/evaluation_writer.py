from prefect.artifacts import create_table_artifact

from kiebids import get_run_logger


class EvaluationWriter:
    def __init__(self):
        self.metrics = {
            "layout-analysis-performance": [],
            "text-recognition-performance": [],
            "semantic-tagging-performance": [],
            "entity-linking-perfomance": [],
        }

    def create_tables(self):
        logger = get_run_logger()
        try:
            for key, value in self.metrics.items():
                create_table_artifact(
                    key=key,
                    table=value,
                    description=f"Evaluation metrics {key}",
                )
        except Exception:
            logger.warning("Failed to create tables for evaluation metrics")
