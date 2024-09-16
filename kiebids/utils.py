import os
import cv2
from pathlib import Path
from prefect.logging import get_logger

from kiebids import config


logger = get_logger(__name__)
logger.setLevel(config.log_level)

def kiebids_wrapper(debug_dir_name="default"):
    # TODO describe what this wrapper does
    def decorator(func):
        def wrapper(*args, **kwargs):
            image = func(*args, **kwargs)

            # writing image to disk in debug mode
            if "debug" in kwargs and kwargs["debug"]:
                debug_path = Path(config.output_path) / debug_dir_name
                os.makedirs(debug_path, exist_ok=True)

                # TODO check for type?
                # TODO check for image_path in kwargs
                image_output_path = debug_path / Path(kwargs["image_path"]).name
                cv2.imwrite(str(image_output_path), image)
                logger.debug("Saved image to: %s", image_output_path)
            return image
        return wrapper

    return decorator

if __name__ == "__main__":
    from kiebids.modules.preprocessing import preprocessing

    preprocessing_output_path = preprocessing(image_path="data/images/raw_image1.png", debug=True)