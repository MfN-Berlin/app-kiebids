import json
import os
import re
from pathlib import Path

import cv2
import fiftyone as fo
import fiftyone.core.labels as fol
import numpy as np
from lxml import etree
from PIL import ImageDraw, ImageFont
from prefect.exceptions import MissingContextError
from prefect.logging import get_logger, get_run_logger

from kiebids import config, fiftyone_dataset

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def get_kiebids_logger(name=""):
    try:
        # This must be inside a prefect context like a @task or a @flow
        logger = get_run_logger()
    except MissingContextError:
        logger = get_logger(name)

    return logger


def debug_writer(debug_path="", module=""):
    """
    Decorator to write outputs of different stages/modules to disk in debug mode.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # If we're not in debug mode, don't do anything
            if config.mode != "debug":
                return func(*args, **kwargs)

            if not debug_path:
                raise ValueError("Debug path not provided")
            if not module:
                raise ValueError("Module not provided")

            logger = get_kiebids_logger(module)

            if not os.path.exists(debug_path):
                os.makedirs(debug_path, exist_ok=True)

            current_image = kwargs.get("current_image_name")

            if module == "preprocessing":
                # add original image to fiftyone dataset
                if fiftyone_dataset is not None:
                    sample = fo.Sample(
                        filepath=f"{Path(config.image_path) / current_image}",
                        tags=["original"],
                    )
                    sample["image_name"] = current_image
                    fiftyone_dataset.add_sample(sample)

                image = func(*args, **kwargs)

                filename = Path(current_image).with_suffix(".jpg")
                image_output_path = Path(debug_path) / filename
                cv2.imwrite(str(image_output_path), image)
                logger.debug("Saved preprocessed image to: %s", image_output_path)

                # add preprocessed image to fiftyone dataset
                if fiftyone_dataset is not None:
                    sample = fo.Sample(
                        filepath=f"{image_output_path}", tags=["preprocessed"]
                    )
                    sample["image_name"] = current_image
                    fiftyone_dataset.add_sample(sample)

                return image
            elif module == "layout_analysis":
                label_masks = func(*args, **kwargs)

                image = kwargs.get("image")

                filename = Path(current_image).stem
                crop_and_save_detections(image, label_masks, filename, debug_path)

                mask_path = Path(debug_path) / "masks"
                os.makedirs(mask_path, exist_ok=True)
                for i, mask in enumerate(label_masks):
                    [x, y, w, h] = mask["bbox"]
                    binary_mask = np.array(
                        mask["segmentation"].copy() * 1, dtype=np.uint8
                    )

                    cv2.rectangle(binary_mask, (x, y), (x + w, y + h), 255, thickness=3)
                    cv2.imwrite(
                        f"{mask_path}/{filename}_mask{i}.jpg", binary_mask * 100
                    )

                if fiftyone_dataset is not None:
                    # Adding detections to the dataset
                    image_output_path = Path(config.image_path) / current_image
                    sample = fo.Sample(
                        filepath=f"{image_output_path}", tags=["layout_analysis"]
                    )
                    sample["image_name"] = current_image
                    sample["predictions"] = fol.Detections(
                        detections=[
                            fol.Detection(
                                label="predicted_object",
                                bounding_box=d["normalized_bbox"],
                            )
                            for d in label_masks
                        ]
                    )
                    fiftyone_dataset.add_sample(sample)

                return label_masks
            elif module == "text_recognition":
                texts_and_bb = func(*args, **kwargs)

                output = {
                    "image_index": kwargs.get("current_image_index"),
                    "regions": texts_and_bb,
                }

                output_path = os.path.join(
                    debug_path, current_image.split(".")[0] + ".json"
                )
                with open(output_path, "w") as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)
                logger.debug("Saved extracted text to: %s", output_path)
                return texts_and_bb
            elif module == "entity_linking":
                return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def crop_image(image: np.array, bounding_box: list[int]):
    """
    Get the cropped image from bounding boxes.

    Args:
        image: the original image as a numpy array (height, width, 3)
        bounding_box: coordinates to crop [x_min, y_min, width,height]
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

        logger.debug("Saved bounding box image to %s", output_path)


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
    """Clear all datasets from the FiftyOne database."""
    datasets = fo.list_datasets()

    for dataset_name in datasets:
        fo.delete_dataset(dataset_name)


def extract_polygon(coordinates):
    return [tuple(map(int, point.split(","))) for point in coordinates.split()]


def bounding_box_to_coordinates(bounding_box: list[int]):
    """
    Convert a bounding box to coordinates.
    params bounding_box:  [x_min, y_min, width, height]

    returns: "x_min,y_min x_max,y_min x_max,y_max x_min,y_max"
    """
    x, y, w, h = bounding_box
    return f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"


def resize(img, max_size):
    h, w = img.shape[:2]
    if max(w, h) > max_size:
        aspect_ratio = h / w
        if w >= h:
            resized_img = cv2.resize(img, (max_size, int(max_size * aspect_ratio)))
        else:
            resized_img = cv2.resize(img, (int(max_size * aspect_ratio), max_size))
        return resized_img
    return img


def read_xml(file_path: str) -> dict:
    """
    Parses an XML file and extracts information about pages, text regions, and text lines.

    Args:
        file_path (str): The path to the XML file to be parsed.

    Returns:
        dict: A dictionary containing the extracted information with the following structure:
            {
                "image_filename": str,  # The filename of the image associated with the page
                "image_width": str,     # The width of the image
                "image_height": str,    # The height of the image
                "text_regions": [       # A list of text regions
                    {
                        "id": str,           # The ID of the text region
                        "orientation": str,  # The orientation of the text region
                        "coords": str,       # The coordinates of the text region
                        "text": str,         # The text content of the whole text region
                        "text_lines": [      # A list of text lines within the text region
                            {
                                "id": str,        # The ID of the text line
                                "coords": str,    # The coordinates of the text line
                                "baseline": str,  # The baseline coordinates of the text line
                                "text": str       # The text content of the text line
                            }
                        ]
                    }
                ]
            }
    """

    tree = etree.parse(file_path)  # noqa: S320  # Using `lxml` to parse untrusted data is known to be vulnerable to XML attacks
    ns = {"ns": tree.getroot().nsmap.get(None, "")}

    page = tree.find(".//ns:Page", namespaces=ns)
    output = {
        "image_filename": page.get("imageFilename"),
        "image_width": page.get("imageWidth"),
        "image_height": page.get("imageHeight"),
        "text_regions": [],
    }

    for region in page.findall(".//ns:TextRegion", namespaces=ns):
        text_region = {
            "id": region.get("id"),
            "orientation": region.get("orientation"),
            "coords": region.find(".//ns:Coords", namespaces=ns).get("points"),
            "text": (
                region.findall(".//ns:TextEquiv", namespaces=ns)[-1]
                .find(".//ns:Unicode", namespaces=ns)
                .text
                if region.findall(".//ns:TextEquiv", namespaces=ns)
                else ""
            ),
            "text_lines": [],
        }

        for line in region.findall(".//ns:TextLine", namespaces=ns):
            if "custom" in line.attrib:
                custom_attributes = line.attrib["custom"]
                matches = re.findall(r"(\w+)\s*\{([^}]*)\}", custom_attributes)
                custom_attributes = [
                    (tag, position.strip()) for tag, position in matches
                ]

            text_region["text_lines"].append(
                {
                    "id": line.get("id"),
                    "coords": line.find(".//ns:Coords", namespaces=ns).get("points"),
                    "baseline": line.find(".//ns:Baseline", namespaces=ns).get(
                        "points"
                    ),
                    "text": (
                        line.find(".//ns:TextEquiv", namespaces=ns)
                        .find(".//ns:Unicode", namespaces=ns)
                        .text
                        or ""
                    ),
                    "custom_attributes": custom_attributes,
                }
            )

        output["text_regions"].append(text_region)

    return output


def get_ground_truth_data(filename):
    xml_file = filename.replace(filename.split(".")[-1], "xml")

    # check if ground truth is available
    if xml_file in os.listdir(config.evaluation.xml_path):
        file_path = os.path.join(config.evaluation.xml_path, xml_file)
        return read_xml(file_path)

    return None
