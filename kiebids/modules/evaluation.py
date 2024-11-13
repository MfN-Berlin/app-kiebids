import os
import csv

import editdistance
from itertools import permutations
from lxml import etree

from kiebids import config, get_logger
from kiebids.utils import extract_polygon
from kiebids.parser import get_ground_truth_text

logger = get_logger(__name__)
logger.setLevel(config.log_level)


class TextEvaluator:
    """
    Class to evaluate the text recognition performance of a model using the Character Error Rate (CER)
    with leveinshtein distance.
    """

    def __init__(self, ground_truth, predictions):
        """
        :param ground_truth: List of ground truth strings.
        :param predictions: List of predicted strings.
        """
        # order predictions to minimize the total edit distance
        if len(ground_truth) == len(predictions):
            self.ground_truth = ground_truth
            self.predictions = self.order_prediction(predictions, ground_truth)
        else:
            self.ground_truth = "".join(ground_truth)
            self.predictions = self.concatenate_to_match(predictions, self.ground_truth)

    def calculate_cer(self, ground_truth, prediction):
        """
        Calculate the Character Error Rate (CER) between a ground truth string and a predicted string.

        :param gt: Ground truth string.
        :param pred: Predicted string.
        :return: CER value.
        """
        distance = editdistance.eval(ground_truth, prediction)

        if len(ground_truth) > 0:
            cer = distance / len(ground_truth)
        elif distance == 0:  # Cover for the case when both strings are empty
            cer = 0
        else:  # Cover for the case when ground truth is empty but prediction is not
            cer = 1
        return float(cer)

    def evaluate(self):
        """
        Evaluate the CER for all ground truth and prediction pairs.

        :return: List of CER values.
        """
        cer_values = [
            self.calculate_cer(gt, pred)
            for gt, pred in zip(self.ground_truth, self.predictions, strict=False)
        ]
        return cer_values

    def average_cer(self):
        """
        Calculate the average CER over all ground truth and prediction pairs.

        :return: Average CER value.
        """
        cer_values = self.evaluate()
        avg_cer = sum(cer_values) / len(cer_values) if cer_values else float("inf")
        return avg_cer

    def order_prediction(self, predictions, ground_truth):
        """
        Orders predictions to minimize the total edit distance.

        :param predictions: List of predicted text.
        :param  ground_truth: List of ground truth.
        :return: Ordered predictions
        """
        if len(predictions) != len(ground_truth):
            return None

        min_distance = float("inf")
        best_permutation = ground_truth

        for perm in permutations(predictions):
            total_distance = sum(
                editdistance.eval(a, b) for a, b in zip(ground_truth, perm)
            )
            if total_distance < min_distance:
                min_distance = total_distance
                best_permutation = perm

        return list(best_permutation)

    def concatenate_to_match(self, target, strings):
        """
        Concatenates a list of strings in an order that makes them as similar as possible to a target string.

        :param target: The target string to match.
        :param strings: List of strings to concatenate.
        :return: Concatenated string that is most similar to the target string.
        """
        min_distance = float("inf")
        best_concatenation = None

        for perm in permutations(strings):
            concatenated = "".join(perm)
            distance = editdistance.eval(target, concatenated)
            if distance < min_distance:
                min_distance = distance
                best_concatenation = concatenated

        return best_concatenation


def evaluate_module(evaluation_path="", module=""):
    """
    Decorator to evaluate the performance of a module
    """
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path, exist_ok=True)
        with open(
            os.path.join(evaluation_path, "Evaluation.csv"), mode="w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "CER"])

    def decorator(func):
        def wrapper(*args, **kwargs):
            current_image_name = kwargs.get("current_image_name")
            if not module:
                return func(*args, **kwargs)

            if module == "layout_analysis":
                bb_labels = func(*args, **kwargs)
                # comparing labels with ground truth
                return bb_labels
            elif module == "text_recognition":
                texts_and_bb = func(*args, **kwargs)
                predictions = [text["text"] for text in texts_and_bb]
                ground_truth = get_ground_truth_text(
                    current_image_name, config.xml_path
                )

                text_evaluator = TextEvaluator(ground_truth, predictions)
                avg_cer = text_evaluator.average_cer()
                logger.info("Average CER: %s", avg_cer)
                with open(
                    os.path.join(evaluation_path, "Evaluation.csv"),
                    mode="a",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow([current_image_name, avg_cer])
                return avg_cer

            elif module == "semantic_labeling":
                # do something here
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_ground_truth(filename):
    xml_file = filename.replace(filename.split(".")[-1], "xml")
    polygons = []

    # check if ground truth is available
    for ds in config.evaluation_datasets:
        if xml_file in os.listdir(config.evaluation_datasets[ds].xml_path):
            # get labels from xml file
            file_path = os.path.join(config.evaluation_datasets[ds].xml_path, xml_file)
            tree = etree.parse(file_path)  # noqa: S320
            root = tree.getroot()
            ns = {"ns": root.nsmap[None]} if None in root.nsmap else {}

            # transcriptions = ""
            textlines = root.xpath(
                "//ns:TextLine" if ns else "//TextLine", namespaces=ns
            )
            for textline in textlines:
                coords = textline.find("ns:Coords" if ns else "Coords", namespaces=ns)
                if coords is not None:
                    polygons.append(extract_polygon(coords.get("points")))

                # unicode_elem = textline.find(".//ns:Unicode" if ns else ".//Unicode", namespaces=ns)
                # if unicode_elem is not None:
                #     transcriptions += f"{i+1}. {unicode_elem.text}\n"

    return polygons
