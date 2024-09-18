import pytesseract
from prefect import task

from kiebids import config, pipeline_config, get_logger

module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = "" if config.mode != "debug" else f"{pipeline_config['debug_path']}/{module}"
module_config = pipeline_config[module]


@task
def text_recognition(image, bb_labels):
    """
    Recognize text from cropped images.
    param:
    image:
    bb_labels:
    """

    tc_results = []
    for bbox in bb_labels:
        # create snippet of image out of bounding box
        x, y, w, h = bbox["bbox"]

        # Crop the image using the bounding box
        cropped_image = image[y : y + h, x : x + w]
        tc_results.append({"text": pytesseract.image_to_string(cropped_image), "bbox": bbox["bbox"]})

    return tc_results
