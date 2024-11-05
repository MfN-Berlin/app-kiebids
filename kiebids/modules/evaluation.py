import os
from io import BytesIO

import editdistance
import requests
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from kiebids import config, get_logger
from kiebids.utils import draw_polygon_on_image

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
            self.calculate_cer(gt, pred) for gt, pred in zip(self.ground_truth, self.predictions, strict=False)
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


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException:
        logger.exception("Error fetching image from URL")
        return None
    except OSError:
        logger.exception("Error opening image")
        return None
    else:
        return image


def process_xml_files(folder_path, output_path):
    """
    Process XML files in the given folder path and
    save the images with polygons and transcriptions in the output path.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    for filename in tqdm(files, desc="Processing XML files"):
        file_path = os.path.join(folder_path, filename)
        tree = etree.parse(file_path)  # noqa: S320
        root = tree.getroot()
        ns = {"ns": root.nsmap[None]} if None in root.nsmap else {}

        comments = root.find(
            ".//ns:Metadata/ns:Comments" if ns else ".//Metadata/Comments",
            namespaces=ns,
        )
        # excluding some fields without assignment
        comments = dict(item.split("=", 1) for item in comments.text.split(", ") if len(item.split("=", 1)) == 2)

        # loading from url
        image_url = comments.get("imgUrl")
        image = None
        if image_url:
            image = load_image_from_url(image_url)

        # lookup for polygon coordinates and transcriptions
        transcriptions = ""
        textlines = root.xpath("//ns:TextLine" if ns else "//TextLine", namespaces=ns)
        for i, textline in enumerate(textlines):
            coords = textline.find("ns:Coords" if ns else "Coords", namespaces=ns)
            if coords is not None:
                points = coords.get("points")
                image = draw_polygon_on_image(image, points, i + 1)

            unicode_elem = textline.find(".//ns:Unicode" if ns else ".//Unicode", namespaces=ns)
            if unicode_elem is not None:
                transcriptions += f"{i+1}. {unicode_elem.text}\n"

        # Add transcriptions as caption to the image
        font = ImageFont.load_default(size=16)
        caption_height = 50 + (20 * len(transcriptions.splitlines()))
        caption_image = Image.new("RGB", (image.width, caption_height), color="black")
        draw = ImageDraw.Draw(caption_image)
        draw.text((10, 10), transcriptions, fill="white", font=font)

        # Combine the original image with the caption image
        new_image = Image.new("RGB", (image.width, image.height + caption_height))
        new_image.paste(image, (0, 0))
        new_image.paste(caption_image, (0, image.height))

        new_image.save(f"{output_path}/polygons_{filename.replace('.xml', '.jpg')}")
