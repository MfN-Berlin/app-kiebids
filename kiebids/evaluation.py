import itertools
import re
from io import BytesIO
from itertools import permutations

import cv2
import numpy as np
import requests
import spacy
from PIL import Image
from torchmetrics.text import CharErrorRate

from kiebids import config, evaluation_writer, get_logger, pipeline_config
from kiebids.utils import extract_polygon, get_ground_truth_data, resize

logger = get_logger(__name__)


def evaluator(module=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # get ground truth for image
            gt_data = get_ground_truth_data(kwargs.get("current_image_name"))

            # skip evaluation if not enabled or no gt data
            if not config.evaluation or not gt_data:
                return func(*args, **kwargs)

            # logger = get_kiebids_logger(module)
            if module == "layout_analysis":
                bb_labels = func(*args, **kwargs)

                if gt_data:
                    gt_regions = [
                        extract_polygon(tr["coords"])
                        for tr in gt_data.get("text_regions")
                    ]
                    # TODO make this casting safe
                    original_resolution = (
                        int(gt_data.get("image_height")),
                        int(gt_data.get("image_width")),
                    )

                    avg_iou = compare_layouts(
                        bb_labels,
                        gt_regions,
                        original_resolution=original_resolution,
                    )
                    avg_iou["image_name"] = kwargs.get("current_image_name")
                    evaluation_writer.metrics["layout-analysis-performance"].append(
                        avg_iou
                    )
                return bb_labels
            elif module == "text_recognition":
                texts_and_bb = func(*args, **kwargs)

                predictions = [text["text"] for text in texts_and_bb]

                gt_texts = [tr["text"] for tr in gt_data.get("text_regions")]

                # INFO: The ground truth xml files sometimes stores linebreakes as \r\n and sometimes \n.
                # For fair comparison we replace all \r\n with \n
                gt_texts = [
                    text.replace("\r\n", "\n") for text in gt_texts if text is not None
                ]

                cers = compare_texts(
                    predictions=predictions,
                    ground_truths=gt_texts,
                    image_index=kwargs.get("current_image_index"),
                )

                # TODO how to track cer when no match is found?
                for cer in cers:
                    cer["image_name"] = kwargs.get("current_image_name")
                evaluation_writer.metrics["text-recognition-performance"].extend(cers)
                return texts_and_bb
            elif module == "semantic_tagging":
                # only have gt for single exhibit labels (regions). in cases when multiple labels are present, we need a way to map gt region to prediction region at hand
                text, gt_spans = prepare_sem_tag_gt(gt_data)
                # Because we want to evaluate the modules standalone behaviour we evaluate this module on the gt spans
                kwargs["text"] = text

                sequences_and_tags = func(*args, **kwargs)

                sample_spans = [s["span"] for s in gt_spans]
                performance = compare_tags(
                    predictions=sample_spans, ground_truths=gt_spans
                )

                performance["image_name"] = kwargs.get("current_image_name")
                evaluation_writer.metrics["semantic-tagging-performance"].append(
                    performance
                )

                return sequences_and_tags
            elif module == "entity_linking":
                text, gt_spans = prepare_sem_tag_gt(gt_data)
                # Because we want to evaluate the modules standalone behaviour we evaluate this module on the gt spans
                kwargs["entities"] = [s["span"] for s in gt_spans]

                entities_geoname_ids = func(*args, **kwargs)

                # compare with gt geoname ids
                performance = compare_geoname_ids(
                    predictions=entities_geoname_ids,
                    ground_truths=gt_spans,
                )

                performance["image_name"] = kwargs.get("current_image_name")
                evaluation_writer.metrics["entity-linking-perfomance"].append(
                    performance
                )
                return entities_geoname_ids
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def compare_layouts(
    predictions: list,
    ground_truths: list,
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

    # TODO could create a chart of individual ious inside evaluation writer but cant show it in prefect ui

    # account for false positives and false negatives
    num_fp_fn = abs(len(ground_truths) - len(predictions))

    avg_iou = {
        "avg_iou": np.average(np.concatenate((np.array(ious), np.zeros(num_fp_fn))))
    }
    logger.debug(f"average iou: {avg_iou}")
    return avg_iou


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
    Computes the Character Error Rate (CER) ground truth and predicted strings,
    using torchmetric CharErrorRate. https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html.
    It orders the predictionst to the ground truth string to minimize the total edit distance.
    Saves the individual CER values and the average CER value to tensorboard.

    Args:
        ground_truth: List of ground truth strings.
        predictions: List of predicted strings.

    """
    # Only evaluate if the number of ground truth strings matches the number of predictions
    if len(ground_truths) != len(predictions):
        logger.warning(
            "Did not evaluate text in image - the number of found text regions are not the same as in the ground truth XML file."
        )
        return []

    CER_calculator = CharErrorRate()

    # Order the predicted strings to the ground truth strings until finding the best possible match
    min_cer = float("inf")
    for perm in permutations(predictions):
        cer = CER_calculator(perm, ground_truths)
        if cer < min_cer:
            min_cer = cer
            ordered_predictions = perm

    # Calculate CER values for each individual region with the best region match in the ground truth
    cer_values = [
        CER_calculator(prediction, ground_truth)
        for prediction, ground_truth in zip(ordered_predictions, ground_truths)
    ]

    logger.debug(
        "average CER: %s - Individual CER values: %s",
        round(float(min_cer), 4),
        [round(float(value), 4) for value in cer_values],
    )

    cers = [{"cer": float(cer), "bb-index": i} for i, cer in enumerate(cer_values)]

    # Save average CER value to tensorboard
    # evaluation_writer.add_scalar("Text_recognition/_average_CER", min_cer, image_index)
    return cers


def prepare_sem_tag_gt(file_dict):
    """
    Prepares the ground truth data for semantic tagging and entity linking evaluation.
    It extracts the text, tags, and positions from the XML file.
    It concatenates the text lines with a line separator and extracts the tags and positions from custom attributes.
    The function returns the concatenated text and a list of gt attributes.
    """

    line_separator = "\n\n"

    tag_lookup = pipeline_config["semantic_tagging"].tag_lookup
    global_positions = []
    global_tags = []
    # multiple regions possible because of multiple exhibit labels.
    # TODO this is just assuming that there is only one region present in evaluation data set. Do we need a strategy to handle multiple regions?
    for region in file_dict["text_regions"]:
        text = []
        # global offset used to correct posiotion for tags
        global_offset = 0
        for line in region["text_lines"]:
            # extract text lines and concatenate (separator=[line_sep])
            text.append(line["text"])

            # skipping reading order
            global_tags.extend(
                [ca[0] for ca in line["custom_attributes"] if ca[0] in tag_lookup]
            )

            # extract positions from custom attributes
            positions = [
                {k: v for k, v in re.findall(r"(\w+):([^;]+);", ca[1])}
                for ca in line["custom_attributes"]
                if ca[0] in tag_lookup
            ]
            # adding global offset to positions offset
            for p in positions:
                p["offset"] = int(p["offset"]) + global_offset

            global_positions.extend(positions)

            global_offset += len(line["text"]) + len(line_separator)

        text = line_separator.join(text)

    nlp = spacy.load("en_core_web_sm")
    sem_tag_gt = []

    # Create a spaCy doc (tokenized version of the text)
    doc_gold = nlp.make_doc(text)
    for tag, p in zip(global_tags, global_positions):
        sem_tag_gt.append(
            {
                # Use char_span to align character offsets to tokens
                "span": doc_gold.char_span(
                    int(p["offset"]), int(p["offset"]) + int(p["length"]), label=tag
                ),
                "geoname_id": p.get("Geonames"),
            }
        )
    return text, sem_tag_gt


def compare_tags(predictions: list, ground_truths: list):
    gold_set = {
        (s["span"].start_char, s["span"].end_char, s["span"].label_)
        for s in ground_truths
    }
    pred_set = {
        (s.start_char, s.end_char, s.label_) for s in predictions if s is not None
    }

    # TODO can we compare like this?
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision, recall, f1 = compute_performance_metrics(tp, fp, fn)
    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "true-positive": tp,
        "false-positive": fp,
        "false-negative": fn,
    }


def compare_geoname_ids(predictions: list, ground_truths: list):
    geo_tags = pipeline_config["entity_linking"].geoname_tags

    # we are only interested in geoname tags and not None geoname ids
    gt_geo_entities = [
        entity
        for entity in ground_truths
        if entity["span"].label_ in geo_tags and entity["geoname_id"] is not None
    ]
    pred_geo_entities = [
        entity for entity in predictions if entity["span"].label_ in geo_tags
    ]

    tp, fp, fn = 0, 0, 0
    for pred in pred_geo_entities:
        for gt in gt_geo_entities:
            gt_span, gt_geoname_id = (gt["span"], int(gt["geoname_id"]))

            # Prediction have a list of geoname ids
            pred_span, pred_geoname_ids = pred["span"], pred["geoname_ids"]
            pred_geoname_ids = [] if pred_geoname_ids is None else pred_geoname_ids

            # TODO comparisson with strings to match the correct tags sufficient?
            if str(pred_span) == str(gt_span):
                tp += (gt_geoname_id in pred_geoname_ids) * 1
                fp += (gt_geoname_id not in pred_geoname_ids) * 1
                # the fn case never occurs because we are only interested in geoname tags.
                # either there is a gt geoname id in geoname tags or the case is invalid
                # thus there is no case where pred of geonames is present and gt is not

    # Invalid evaluation if no geoname tags are present in gt
    precision, recall, f1 = (
        ("invalid", "invalid", "invalid")
        if len(gt_geo_entities) == 0
        else compute_performance_metrics(tp, fp, fn)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true-positive": tp,
        "false-positive": fp,
        "false-negative": fn,
        "geonames-in-gt": len(gt_geo_entities),
    }


def compute_performance_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2)
