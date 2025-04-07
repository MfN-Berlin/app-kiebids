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


class SemanticTagging:
    def __init__(self):
        logger.info("Running Semantic tagging module")

    @debug_writer(debug_path, module=module)
    @evaluator(module=module)
    @task(name=module)
    def run(self, text, **kwargs):  # pylint: disable=unused-argument
        logger.debug("%s", text)
        return "dummy"
