import os
import yaml
import cv2

from pathlib import Path
from prefect import task
from prefect.logging import get_logger

from kiebids import pipeline_config, config
from kiebids.utils import debug_writer

logger = get_logger(__name__)
logger.setLevel(config.log_level)

debug_path = "preprocessing"
preprocessing_config = pipeline_config["preprocessing"]


@task
@debug_writer(debug_path)
def preprocessing(image_path, debug=False):
    logger.info("Preprocessing image: %s", image_path)
    image = cv2.imread(image_path)

    if preprocessing_config["gray"]["enabled"]:
        image = gray(image)

    if preprocessing_config["smooth"]["enabled"]:
        image = smooth(image)

    if preprocessing_config["threshold"]["enabled"]:
        image = threshold(image)

    if preprocessing_config["denoise"]["enabled"]:
        image = denoise(image)

    if preprocessing_config["contrast"]["enabled"]:
        image = contrast(image)

    return image


def gray(image):
    """Converts an image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def smooth(image):
    """Smoothens an image"""
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed


def threshold(image):
    """Applies thresholding to an image"""
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresholded


def denoise(image):
    """Denoises an image"""
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised


def contrast(image):
    """Increases the contrast of an image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(image)
    return contrasted
