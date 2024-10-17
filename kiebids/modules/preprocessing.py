import cv2
from prefect import task

from kiebids import config, pipeline_config, get_logger, evaluation_writer
from kiebids.utils import debug_writer


module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = "" if config.mode != "debug" else f"{config['debug_path']}/{module}"
module_config = pipeline_config[module]


@debug_writer(debug_path, module=module)
@task(name=module)
def preprocessing(image_path):
    logger.info("Preprocessing image: %s", image_path)
    image = cv2.imread(image_path)

    evaluation_writer.add_image("_original", image.transpose(2, 0, 1), 0)

    # TODO resizing only when needed
    # scale = 1/6
    # down_points = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    # image = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

    if module_config["gray"]:
        image = gray(image)

    if module_config["smooth"]:
        image = smooth(image)

    if module_config["threshold"]:
        image = threshold(image)

    if module_config["denoise"]:
        image = denoise(image)

    if module_config["contrast"]:
        image = contrast(image)

    return image


def gray(image):
    """Converts an image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def smooth(image):
    """Smoothens an image"""
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed


def threshold(image):
    """Applies thresholding to an image"""
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
