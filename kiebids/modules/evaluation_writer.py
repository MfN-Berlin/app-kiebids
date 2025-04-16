from prefect.artifacts import create_table_artifact

# circular import
# from kiebids.utils import (
#     get_kiebids_logger
# )


class EvaluationWriter:
    def __init__(self):
        # self.logger = get_kiebids_logger("evaluation")
        # self.logger.info("Running evaluation module")
        self.entity_linking_perfomance = []
        self.layout_analysis_performance = []
        self.text_recognition_performance = []

    def create_table(self):
        try:
            create_table_artifact(
                # TODO naming of the artifact. keys have weird restriction so that file names wont work
                # key=f"{kwargs.get('current_image_name')}",
                key="entity-linking-performance",
                table=self.entity_linking_perfomance,
                description="Performance metrics for geoname ids",
            )
            create_table_artifact(
                key="text-recognition-performance",
                table=self.text_recognition_performance,
                description="text recognition performance",
            )
        except Exception:
            pass
            # self.logger.warning(
            #     "Failed to create artifact for entity linking performance metrics"
            # )
