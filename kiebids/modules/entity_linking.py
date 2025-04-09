import os

import requests
from prefect import task

from kiebids import config, get_logger, pipeline_config, run_id
from kiebids.modules.evaluation import evaluator
from kiebids.utils import debug_writer

module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


class EntityLinking:
    def __init__(self):
        logger.info("Running %s module", module)

    @debug_writer(debug_path, module=module)
    @evaluator(module=module)
    @task(name=module)
    def run(self, entities, **kwargs):  # pylint: disable=unused-argument
        entities_geoname_ids = []
        for entity in entities:
            if entity.label_.startswith("MfN_Geo"):
                api_params = {
                    "q": str(entity),
                    "fuzzy": 0.8,
                    "username": os.getenv("GEONAMES_API_USERNAME"),
                }
                try:
                    response = requests.get(
                        module_config.geonames_api_url, params=api_params, timeout=60
                    )
                    response.raise_for_status()

                    results_list = response.json()["geonames"]
                    # TODO which element in the response list should we take? taking the first one for now
                    if results_list:
                        entities_geoname_ids.append(
                            (entity, results_list[0]["geonameId"])
                        )

                except requests.exceptions.HTTPError as e:
                    logger.info(f"Request error: {e}")

        return entities_geoname_ids


if __name__ == "__main__":
    from kiebids.modules.evaluation import prepare_sem_tag_gt
    from kiebids.utils import get_ground_truth_data

    module_config = pipeline_config["entity_linking"]

    file = "0011_20230207T120422_d42fda_fc542f9f-d7d2-4b48-a2c9-0ab8ad9b8cae_label_front_0001_label.xml"
    parsed_dict = get_ground_truth_data(file)
    text, gt_spans = prepare_sem_tag_gt(parsed_dict)

    entity_linking = EntityLinking()

    nearby_streets = entity_linking.run(
        entities=gt_spans,
        current_image_name=file,
    )

    if nearby_streets:
        for street in nearby_streets.get("streetSegment", []):
            print(street.get("name"))
