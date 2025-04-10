from prefect import task

from kiebids import config, pipeline_config, run_id
from kiebids.modules.evaluation import evaluator
from kiebids.utils import debug_writer, get_kiebids_logger

module = __name__.split(".")[-1]

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


class SemanticTagging:
    def __init__(self):
        self.logger = get_kiebids_logger(module)
        self.logger.info("Running Semantic tagging module")

    @task(name=module)
    @debug_writer(debug_path, module=module)
    @evaluator(module=module)
    def run(self, text, **kwargs):  # pylint: disable=unused-argument
        self.logger.debug("%s", text)
        return "dummy"
