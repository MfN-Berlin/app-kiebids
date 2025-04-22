import os

import requests
from prefect import task

from kiebids import config, pipeline_config, run_id
from kiebids.modules.evaluation import evaluator
from kiebids.utils import debug_writer, get_kiebids_logger

module = __name__.split(".")[-1]

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


class EntityLinking:
    def __init__(self):
        self.logger = get_kiebids_logger(module)
        self.logger.info("Running %s module", module)

    @task(name=module)
    @debug_writer(debug_path, module=module)
    @evaluator(module=module)
    def run(self, entities, **kwargs):  # pylint: disable=unused-argument
        entities_geoname_ids = []
        for entity in entities:
            if entity.label_ in module_config.geoname_tags:
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
                    if results_list:
                        entities_geoname_ids.append(
                            {
                                "span": entity,
                                # taking the entire list of geoname_ids because it is unclear which one to prefer
                                "geoname_ids": [r["geonameId"] for r in results_list],
                            }
                        )

                except requests.exceptions.HTTPError as e:
                    self.logger.info(f"Request error: {e}")

        return entities_geoname_ids
