import os
import cv2
from pathlib import Path
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

            if "debug" in kwargs and kwargs["debug"]:
                # TODO check for type?
                # TODO check for image_path in kwargs
                image_output_path = debug_path / Path(kwargs["image_path"]).name
                cv2.imwrite(str(image_output_path), image)
                logger.debug("Saved image to: %s", image_output_path)
            return image

        return wrapper

    return decorator
