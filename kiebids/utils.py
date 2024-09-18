import os
from pathlib import Path

import cv2
from prefect.logging import get_logger

from kiebids import config

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def debug_writer(debug_dir_name="default"):
    """
    Decorator to write images outputs to disk in debug mode.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.path.exists(config.output_path):
                debug_path = Path(config.output_path) / debug_dir_name
                os.makedirs(debug_path, exist_ok=True)

            image = func(*args, **kwargs)

            if kwargs.get("debug"):
                # TODO check for type?
                # TODO check for image_path in kwargs
                image_output_path = debug_path / Path(kwargs["image_path"]).name
                cv2.imwrite(str(image_output_path), image)
                logger.debug("Saved image to: %s", image_output_path)
            return image

        return wrapper

    return decorator


def plot_and_save_bbox_images(image, masks, image_name, output_dir):
    """
    Plot and save individual images for each mask, using the bounding box to crop the image.

    Args:
    image (numpy.ndarray): The original image as a numpy array (height, width, 3).
    masks (list): A list of dictionaries, each containing a 'bbox' key with [x, y, width, height].
    output_dir (str): Directory to save the output images.
    """

    for i, mask in enumerate(masks, 1):
        bbox = mask["bbox"]
        x, y, w, h = bbox

        # Crop the image using the bounding box
        cropped_image = image[y : y + h, x : x + w]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"{image_name}_{i}.png")
        cv2.imwrite(output_path, cropped_image)

        logger.info("Saved bounding box image to %s", output_path)
