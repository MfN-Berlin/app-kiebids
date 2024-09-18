import os
from pathlib import Path

import cv2
from prefect.logging import get_logger

from kiebids import config

logger = get_logger(__name__)
logger.setLevel(config.log_level)


# TODO interface for different stages
def debug_writer(debug_path="", module=""):
    """
    Decorator to write outputs of different stages/modules to disk in debug mode.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # When debug path not given, no need to do anything
            if not debug_path:
                return func(*args, **kwargs)

            if not os.path.exists(debug_path):
                os.makedirs(debug_path, exist_ok=True)

            if module == "preprocessing":
                image = func(*args, **kwargs)

                if kwargs.get("image_path"):
                    image_output_path = Path(debug_path) / Path(kwargs["image_path"]).name
                    cv2.imwrite(str(image_output_path), image)
                    logger.debug("Saved image to: %s", image_output_path)
                return image
            elif module == "layout_analysis":
                label_masks = func(*args, **kwargs)
                # TODO make image kwargs
                image_name = "test"
                image = args[1]
                plot_and_save_bbox_images(image, label_masks, image_name, debug_path)

                return label_masks

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
        x, y, w, h = mask["bbox"]

        # Crop the image using the bounding box
        cropped_image = image[y : y + h, x : x + w]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"{image_name}_{i}.png")
        cv2.imwrite(output_path, cropped_image)

        logger.info("Saved bounding box image to %s", output_path)
