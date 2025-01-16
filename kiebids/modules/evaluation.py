import itertools
from io import BytesIO
from itertools import permutations

import cv2
import editdistance
import numpy as np
import requests
from PIL import Image

from kiebids import (
    config,
    evaluation_writer,
    event_accumulator,
    get_logger,
    pipeline_config,
)
from kiebids.utils import extract_polygon, get_ground_truth_data, resize

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def evaluator(module=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # skip evaluation if not enabled
            if not module or not evaluation_writer:
                return func(*args, **kwargs)

            if module == "layout_analysis":
                bb_labels = func(*args, **kwargs)

                # get ground truth for image
                parsed_dict = get_ground_truth_data(kwargs.get("current_image_name"))
                if parsed_dict:
                    gt_regions = [
                        extract_polygon(tr["coords"])
                        for tr in parsed_dict.get("text_regions")
                    ]
                    # TODO make this casting safe
                    original_resolution = (
                        int(parsed_dict.get("image_height")),
                        int(parsed_dict.get("image_width")),
                    )
                    logger.debug(f"image index: {kwargs.get('current_image_index')}")
                    compare_layouts(
                        bb_labels,
                        gt_regions,
                        image_index=kwargs.get("current_image_index"),
                        filename=kwargs.get("current_image_name"),
                        original_resolution=original_resolution,
                    )

                return bb_labels
            elif module == "text_recognition":
                texts_and_bb = func(*args, **kwargs)

                predictions = [text["text"] for text in texts_and_bb]
                parsed_dict = get_ground_truth_data(kwargs.get("current_image_name"))

                if parsed_dict:
                    gt_texts = [tr["text"] for tr in parsed_dict.get("text_regions")]
                    compare_texts(
                        predictions=predictions,
                        ground_truths=gt_texts,
                        image_index=kwargs.get("current_image_index"),
                    )

                return texts_and_bb

            elif module == "semantic_labeling":
                # do something here
                return func(*args, **kwargs)

        return wrapper

    return decorator


def compare_layouts(
    predictions: list,
    ground_truths: list,
    image_index: int,
    filename: str,
    original_resolution: tuple,
):
    """
    Compares predictions with ground truths based on highest iou.
    Creates a confusion matrix with ious as values and gt + pred indices as axis.
    Matches gt with pred based on highest iou.
    If there are too many or too few predictions, the iou is set to 0 for the missing ones.
    Logs the ious per region and the average iou for the whole image to tensorboard.

    :param ground_truths: List of ground truth polygons.
    :param predictions: List of dictionaries containing the predicted bounding boxes.
    """
    # create confusion matrix with ious as values and gt + pred indices as axis
    cm_shape = (
        max(len(ground_truths), len(predictions)),
        max(len(ground_truths), len(predictions)),
    )
    gt_pred_confusion_matrix = np.zeros(cm_shape)
    gt_pred_product = list(
        itertools.product(range(len(ground_truths)), range(len(predictions)))
    )

    for gt_index, pred_index in gt_pred_product:
        # if index is out of bounds leave iou at 0 => not enough preds or too many preds
        if gt_index > (len(ground_truths) - 1) or pred_index > (len(predictions) - 1):
            continue
        else:
            pred_sum = predictions[pred_index]["segmentation"]
            gt_sum = create_polygon_mask(ground_truths[gt_index], original_resolution)
            gt_sum = resize(
                gt_sum, pipeline_config["preprocessing"].max_image_dimension
            )

            iou = compute_iou(pred_sum, gt_sum)
            # update iou to confusion matrix
            gt_pred_confusion_matrix[gt_index, pred_index] = iou

    ious = []
    # get 1 to 1 mapping from max values of iou
    while gt_pred_confusion_matrix.any() and np.max(gt_pred_confusion_matrix) > 0:
        logger.debug(f"max iou: {np.max(gt_pred_confusion_matrix)}")
        max_iou_coordinates = np.unravel_index(
            np.argmax(gt_pred_confusion_matrix), gt_pred_confusion_matrix.shape
        )

        ious.append(gt_pred_confusion_matrix[max_iou_coordinates])

        # get max iou and create smaller conf matrix => match found => go on with next gt and pred
        # remove gt row from confusion matrix => no more matches possible
        gt_pred_confusion_matrix = np.delete(
            gt_pred_confusion_matrix, max_iou_coordinates[0], axis=0
        )

    # Add the ious to tensorboard
    evaluation_writer.add_scalars(
        "Layout_analysis/_ious",
        {f"bb_{i}": iou for i, iou in enumerate(ious)},
        image_index,
    )

    # account for false positives and false negatives
    num_fp_fn = abs(len(ground_truths) - len(predictions))

    # average ious
    avg_iou = np.average(np.concatenate((np.array(ious), np.zeros(num_fp_fn))))
    logger.debug(f"average iou: {avg_iou}")
    evaluation_writer.add_scalar("Layout_analysis/_average_ious", avg_iou, image_index)


def create_polygon_mask(polygon_points, image_shape):
    """
    Creates a mask of the polygon in the given image.

    :param polygon_points: List of (x, y) tuples representing the polygon vertices.
    :param image_shape: Tuple (height, width) representing the image shape.
    :return: A binary mask where the polygon area is filled with 1's, and the rest is 0's.
    """
    height, width = image_shape

    # Create a blank mask (same size as the image, single channel)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert polygon_points to a format accepted by OpenCV (an array of shape Nx1x2)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # Draw the polygon on the mask (fill the polygon with white color - value 1)
    cv2.fillPoly(mask, [polygon_points], 1)

    return mask


def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray):
    """
    computes iou and its weight based on union relative to total num of pixels

    Args:
        prediction (): prediction of model
        ground_truth (): ground truth

    Returns:
        iou:
    """
    intersection = np.count_nonzero(prediction & ground_truth)
    union = np.count_nonzero(prediction | ground_truth)

    # union == 0 should never occur because we must catch this case before calling compute_iou => meaning no prediction and gt
    return np.nan if union == 0 else intersection / union


def load_image_from_url(url):
    try:
        response = requests.get(url)  # noqa: S113
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))

        return image  # noqa: TRY300
    except requests.exceptions.RequestException:
        logger.exception("Error fetching image from URL")
        return None
    except OSError:
        logger.exception("Error opening image")
        return None


def compare_texts(predictions: list[str], ground_truths: list[str], image_index: int):
    """
    Computes the Character Error Rate (CER) with leveinshtein distance between ground truth and predicted strings.
    It orders the predictionst to the ground truth string to minimize the total edit distance.
    Saves the individual CER values and the average CER value to tensorboard.

    Args:
        ground_truth: List of ground truth strings.
        predictions: List of predicted strings.

    """
    # Only evaluate if the number of ground truth strings matches the number of predictions
    if len(ground_truths) != len(predictions):
        event_accumulator.Reload()
        if "Text_recognition/_average_CER" in event_accumulator.scalars.Keys():
            logger.warning(
                "Did not evaluate text in image, len(pred)!=len(ground_truth). Evaluated images in TB: %s/%s",
                len(event_accumulator.Scalars("Text_recognition/_average_CER")),
                len(event_accumulator.Scalars("Layout_analysis/_average_ious")),
            )
        else:
            logger.warning(
                "Did not evaluate text in image, len(pred)!=len(ground_truth)."
            )
        return

    # Order predictions
    min_distance = float("inf")

    for perm in permutations(predictions):
        cer_values_perm = [compute_cer(p, gt) for p, gt in zip(perm, ground_truths)]

        if sum(cer_values_perm) < min_distance:
            min_distance = sum(cer_values_perm)
            cer_values = cer_values_perm

    # Save individual CER values to tensorboard
    evaluation_writer.add_scalars(
        "Text_recognition/_CER",
        {f"bb_{i}": cer for i, cer in enumerate(cer_values)},
        image_index,
    )

    avg_cer = np.average(cer_values)

    logger.debug("average CER: %s - cer values: %s", avg_cer, cer_values)

    # Save average CER value to tensorboard
    evaluation_writer.add_scalar("Text_recognition/_average_CER", avg_cer, image_index)


def compute_cer(prediction: str, ground_truth: str):
    """
    Computes the Character Error Rate (CER) with leveinshtein distance between ground truth and predicted strings.

    Args:
        prediction: Predicted string.
        ground_truth: Ground truth string.
    """
    distance = editdistance.eval(ground_truth, prediction)

    if len(ground_truth) > 0:
        cer_value = distance / len(ground_truth)
    # Cover for the case when both strings are empty
    elif distance == 0:
        cer_value = 0
    # Cover for the case when ground truth is empty but prediction is not
    else:
        cer_value = len(prediction)
    return float(cer_value)
