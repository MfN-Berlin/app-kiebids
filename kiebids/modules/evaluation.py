import os

import editdistance
from lxml import etree

from kiebids import config, get_logger
from kiebids.utils import extract_polygon

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def evaluate_module(module=""):
    """ """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not module:
                return func(*args, **kwargs)

            if module == "layout_analysis":
                bb_labels = func(*args, **kwargs)
                # comparing labels with ground truth
                return bb_labels
            elif module == "text_recognition":
                text_and_labels = func(*args, **kwargs)
                # comparing text with ground truth
                return text_and_labels
            elif module == "semantic_labeling":
                # do something here
                return func(*args, **kwargs)

        return wrapper

    return decorator


class TextEvaluator:
    """
    Class to evaluate the text recognition performance of a model using the Character Error Rate (CER)
    with leveinshtein distance.
    """

    def __init__(self, ground_truths, predictions):
        """
        :param ground_truth: List of ground truth strings.
        :param predictions: List of predicted strings.
        """
        self.ground_truth = ground_truths
        self.predictions = predictions

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
