import os

import pytesseract

import cv2

from pathlib import Path
from prefect import task


@task
def text_recognition(input_dir, output_path, debug=False):
    """
    Recognize text from cropped images.
    param:
    input_path: str, path to directory containing cropped images
    output_path: str, path to output directory
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
