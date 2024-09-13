import os
import yaml
import cv2

from pathlib import Path
from prefect import task

from kiebids import pipeline_config


# Import config
script_path = Path(__file__).parent.parent.resolve()

preprocessing_config = pipeline_config["preprocessing"]

# Create debugging path
DEBUG_PATH = script_path.parent / "data" / "debug" / "preprocessing"
os.makedirs(DEBUG_PATH, exist_ok=True)


@task
def preprocessing(image_path, output_path, debug=False):
    OUTPUT_DIR_PREPROCESSING = Path(output_path) / "preprocessing"
    os.makedirs(OUTPUT_DIR_PREPROCESSING, exist_ok=True)

    # Todo: Change prints to loggings
    print("Preprocessing image: ", image_path)
    print("Debug mode: ", debug)
    image = cv2.imread(image_path)

    if preprocessing_config["gray"]["enabled"]:
        image = gray(image, debug=debug)

    if preprocessing_config["smooth"]["enabled"]:
        image = smooth(image, debug=debug)

    if preprocessing_config["threshold"]["enabled"]:
        image = threshold(image, debug=debug)

    if preprocessing_config["denoise"]["enabled"]:
        image = denoise(image, debug=debug)

    if preprocessing_config["contrast"]["enabled"]:
        image = contrast(image, debug=debug)

    # Save the image
    image_output_path = OUTPUT_DIR_PREPROCESSING / Path(image_path).name
    cv2.imwrite(str(image_output_path), image)
    return str(image_output_path)


def gray(image, debug=False):
    """Converts an image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug is True:
        image_name = DEBUG_PATH / "gray_image.jpg"
        print("Saving gray image to: ", image_name)
        cv2.imwrite(str(image_name), gray)
    return gray


def smooth(image, debug=False):
    """Smoothens an image"""
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    if debug:
        image_name = DEBUG_PATH / "smoothed_image.jpg"
        cv2.imwrite(str(image_name), smoothed)
    return smoothed


def threshold(image, debug=False):
    """Applies thresholding to an image"""
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if debug:
        image_name = DEBUG_PATH / "thresholded_image.jpg"
        cv2.imwrite(str(image_name), thresholded)
    return thresholded


def denoise(image, debug=False):
    """Denoises an image"""
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    if debug:
        image_name = DEBUG_PATH / "denoised_image.jpg"
        cv2.imwrite(str(image_name), denoised)
    return denoised


def contrast(image, debug=False):
    """Increases the contrast of an image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(image)
    if debug:
        image_name = DEBUG_PATH / "contrasted_image.jpg"
        cv2.imwrite(str(image_name), contrasted)
    return contrasted
