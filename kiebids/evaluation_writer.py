from prefect.artifacts import create_table_artifact

from kiebids import get_run_logger


class EvaluationWriter:
    def __init__(self):
        self.init_metrics()

    def init_metrics(self):
        """Reinitialize metrics for the next run."""
        self.metrics = {
            "layout-analysis-performance": [],
            "text-recognition-performance": [],
            "semantic-tagging-performance": [],
            "entity-linking-perfomance": [],
        }

    def create_tables(self):
        """Create tables for evaluation metrics on all processed images."""
        logger = get_run_logger()
        try:
            for key, data in self.metrics.items():
                create_table_artifact(
                    key=key,
                    table=data,
                    description=f"Evaluation metrics {key}",
                )
        except Exception:
            logger.warning("Failed to create tables for evaluation metrics")

    def create_table(self):
        """Create table for evaluation metrics from most recent run."""
        logger = get_run_logger()
        try:
            for key, data in self.metrics.items():
                if data:
                    create_table_artifact(
                        key=key,
                        table=[data[-1]],
                        description=f"Evaluation metrics {key}",
                    )
        except Exception as e:
            logger.warning("Failed to create tables for evaluation metrics. %s", e)
