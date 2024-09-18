import os
from pathlib import Path

import cv2
import pytesseract
from prefect import task
from prefect.logging import get_logger

from kiebids import config, pipeline_config

module = "layout_analysis"
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = f"{pipeline_config['debug_path']}/{module}"
module_config = pipeline_config[module]


@task
def text_recognition(image, bb_labels, debug=False):
    """
    Recognize text from cropped images.
    param:
    image:
    bb_labels:
    """

    OUTPUT_DIR_TEXT_RECOGNITION = Path(output_path) / "text_recogniton"
    os.makedirs(OUTPUT_DIR_TEXT_RECOGNITION, exist_ok=True)

    # Get all cropped images related to the input image
    images = [image for image in os.listdir(input_dir)]

    for image in images:
        image_path = Path(input_dir) / image
        text = get_text(image_path)

        image_name = image.split(".")[0] + ".txt"
        text_output_path = OUTPUT_DIR_TEXT_RECOGNITION / image_name

        with open(text_output_path, "w") as f:
            f.write(text)

    return str(OUTPUT_DIR_TEXT_RECOGNITION)


def get_text(image_path, debug=False):
    """
    Get ocr text from image with tesseract
    """
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text
