import json
import os
from pathlib import Path
import fiftyone as fo
import fiftyone.core.labels as fol

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont
from prefect.logging import get_logger

from kiebids import config, current_dataset

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def debug_writer(debug_path="", module=""):
    """
    Decorator to write outputs of different stages/modules to disk in debug mode.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # When debug path not given, no need to do anything
            if not debug_path or not module:
                return func(*args, **kwargs)

            if not os.path.exists(debug_path):
                os.makedirs(debug_path, exist_ok=True)

            current_image = kwargs.get("current_image_name")

            if module == "preprocessing":
                # add original image to dataset
                sample = fo.Sample(filepath=f"{Path(config.image_path) / current_image}", tags=["original"])
                sample["image_name"] = current_image
                current_dataset.add_sample(sample)

                image = func(*args, **kwargs)

                image_output_path = Path(debug_path) / current_image
                cv2.imwrite(str(image_output_path), image)
                logger.debug("Saved preprocessed image to: %s", image_output_path)

                # add preprocessed image to fiftyone dataset
                sample = fo.Sample(filepath=f"{image_output_path}", tags=["preprocessed"])
                sample["image_name"] = current_image
                current_dataset.add_sample(sample)

                return image
            elif module == "layout_analysis":
                label_masks = func(*args, **kwargs)

                image = kwargs.get("image")

                # TODO are the crops still needed somewhere?
                crop_and_save_detections(image, label_masks, current_image.split(".")[0], debug_path)

                # Adding detections to the dataset
                image_output_path = Path(config.image_path) / current_image
                sample = fo.Sample(filepath=f"{image_output_path}", tags=["layout_analysis"])
                sample["image_name"] = current_image
                sample["predictions"] = fol.Detections(
                    detections=[
                        fol.Detection(label="predicted_object", bounding_box=d["normalized_bbox"]) for d in label_masks
                    ]
                )

                current_dataset.add_sample(sample)

                return label_masks
            elif module == "text_recognition":
                texts = func(*args, **kwargs)

                output_path = os.path.join(debug_path, current_image.split(".")[0] + ".json")
                with open(output_path, "w") as f:
                    json.dump(texts, f, ensure_ascii=False, indent=4)
                logger.debug("Saved extracted text to: %s", output_path)
                return texts

        return wrapper

    return decorator


def crop_image(image: np.array, bounding_box: list[int]):
    """get the cropped image from bounding boxes.
    Parameters:
        image: he original image as a numpy array (height, width, 3)
        bounding_box: coordinates to crop [x_min,y_min,width,height]
    """
    x, y, w, h = bounding_box
    return image[y : y + h, x : x + w]


def crop_and_save_detections(image, masks, image_name, output_dir):
    """
    Plot and save individual images for each mask, using the bounding box to crop the image.

    Args:
    image (numpy.ndarray): The original image as a numpy array (height, width, 3).
    masks (list): A list of dictionaries, each containing a 'bbox' key with [x, y, width, height].
    output_dir (str): Directory to save the output images.
    """

    for i, mask in enumerate(masks, 1):

        # Crop the image using the bounding box
        cropped_image = crop_image(image=image, bounding_box=mask["bbox"])

        # Save the cropped image
        output_path = os.path.join(output_dir, f"{image_name}_{i}.png")
        cv2.imwrite(output_path, cropped_image)

        logger.info("Saved bounding box image to %s", output_path)


def draw_polygon_on_image(image, coordinates, i=-1):
    draw = ImageDraw.Draw(image)
    points = [tuple(map(int, point.split(","))) for point in coordinates.split()]
    draw.polygon(points, outline="red", fill=None, width=2)

    if i >= 0:
        # Calculate the upper-left corner for the label
        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)

        label_position = (x_min, y_min - 10)
        font = ImageFont.load_default(size=24)
        draw.text(label_position, str(i), fill="blue", font=font)

    return image


def clear_fiftyone():
    """
    Clear all datasets from the FiftyOne database.
    """
    datasets = fo.list_datasets()

    for dataset_name in datasets:
        fo.delete_dataset(dataset_name)


def resize(img, max_size):
    h, w, _ = img.shape
    if max(w, h) > max_size:
        aspect_ratio = h / w
        if w >= h:
            resized_img = cv2.resize(img, (max_size, int(max_size * aspect_ratio)))
        else:
            resized_img = cv2.resize(img, (int(max_size * aspect_ratio), max_size))
        return resized_img
    return img
