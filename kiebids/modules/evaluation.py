import os
from io import BytesIO
import itertools

import cv2
import numpy as np
import requests
from lxml import etree
from PIL import Image
from tqdm import tqdm

from kiebids import config, evaluation_writer, get_logger
from kiebids.utils import extract_polygon

logger = get_logger(__name__)
logger.setLevel(config.log_level)


def evaluator(module=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # skip evaluation if not enabled
            if not module or not config.evaluation:
                return func(*args, **kwargs)

            if module == "layout_analysis":
                bb_labels = func(*args, **kwargs)
                # get ground truth for image
                gt_labels = get_ground_truth(kwargs.get("current_image_name"))
                if gt_labels:
                    compare_layouts(
                        bb_labels, gt_labels, kwargs.get("current_image_name")
                    )

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


def compare_layouts(predictions: list, ground_truths: list, filename: str):
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
            gt_sum = create_polygon_mask(ground_truths[gt_index], pred_sum.shape)

            # Log the image to TensorBoard
            # padding = np.ones((gt_sum.shape[0], 50), dtype=np.uint8) * 255
            # combined_image = np.concatenate(
            #     [gt_sum * 150, padding, pred_sum * 150], axis=1
            # )
            # evaluation_writer.add_image(
            #     f"{filename}-gt-left_pred-right",
            #     combined_image[np.newaxis, ...],
            #     i * len(ground_truth) + j,
            # )

            iou, _ = compute_iou(pred_sum, gt_sum)
            # update iou to confusion matrix
            gt_pred_confusion_matrix[gt_index, pred_index] = iou

    ious = []
    # get 1 to 1 mapping from max values of iou
    while np.max(gt_pred_confusion_matrix) > 0:
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

    # account for false positives and false negatives
    num_fp_fn = abs(len(ground_truths) - len(predictions))

    # average ious
    avg_iou = np.average(np.concatenate((np.array(ious), np.zeros(num_fp_fn))))
    logger.debug(f"average iou: {avg_iou}")
    evaluation_writer.add_scalar("_average_ious", avg_iou)

    evaluation_writer.flush()


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
        weight: weight of the iou
    """
    intersection = np.count_nonzero(prediction & ground_truth)
    union = np.count_nonzero(prediction | ground_truth)

    # with this weighting we punish cases where pred is much bigger than gt
    weight = union / ground_truth.size
    # union == 0 should never occur because we must catching this case before calling compute_iou
    iou = np.nan if union == 0 else intersection / union

    return iou, weight


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


def process_xml_files(folder_path, output_path):
    """
    Process XML files in the given folder path and
    save the images with polygons and transcriptions in the output path.
    """

    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    for filename in tqdm(files[:10], desc="Processing XML files"):
        file_path = os.path.join(folder_path, filename)
        tree = etree.parse(file_path)  # noqa: S320
        root = tree.getroot()
        ns = {"ns": root.nsmap[None]} if None in root.nsmap else {}

        comments = root.find(
            ".//ns:Metadata/ns:Comments" if ns else ".//Metadata/Comments",
            namespaces=ns,
        )
        # excluding some fields without assignment
        comments = dict(
            item.split("=", 1)
            for item in comments.text.split(", ")
            if len(item.split("=", 1)) == 2
        )

        # loading from url
        image_url = comments.get("imgUrl")
        image = None
        if image_url:
            image = load_image_from_url(image_url)
            grayscale_image = image.convert("L")

            kernel = np.ones((5, 5), np.uint8)
            # Convert the grayscale PIL image to a NumPy array (of type uint8)
            grayscale_np = np.array(grayscale_image, dtype=np.uint8)

            cv2.imwrite(
                f"{output_path}/{filename.replace('.xml', '_gs.jpg')}", grayscale_np
            )

            # Apply adaptive thresholding using cv2.adaptiveThreshold
            thresholded_image = cv2.adaptiveThreshold(
                grayscale_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2
            )

            ret, binary = cv2.threshold(grayscale_np, 100, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # up_points = (grayscale_np.shape[0] * 4, grayscale_np.shape[1] * 4)
            # resized_up = cv2.resize(binary, up_points, interpolation=cv2.INTER_LINEAR)

            image.save(f"{output_path}/{filename.replace('.xml', '_orig.jpg')}")
            cv2.imwrite(
                f"{output_path}/{filename.replace('.xml', '_bw.jpg')}",
                thresholded_image,
            )
            cv2.imwrite(
                f"{output_path}/{filename.replace('.xml', '_open.jpg')}", opening
            )
            cv2.imwrite(
                f"{output_path}/{filename.replace('.xml', '_closing.jpg')}", closing
            )
            cv2.imwrite(
                f"{output_path}/{filename.replace('.xml', '_binary.jpg')}", binary
            )

        # # lookup for polygon coordinates and transcriptions
        # transcriptions = ""
        # textlines = root.xpath("//ns:TextLine" if ns else "//TextLine", namespaces=ns)
        # for i, textline in enumerate(textlines):
        #     coords = textline.find("ns:Coords" if ns else "Coords", namespaces=ns)
        #     if coords is not None:
        #         points = coords.get("points")
        #         image = draw_polygon_on_image(image, points, i + 1)

        #     unicode_elem = textline.find(".//ns:Unicode" if ns else ".//Unicode", namespaces=ns)
        #     if unicode_elem is not None:
        #         transcriptions += f"{i+1}. {unicode_elem.text}\n"

        # # Add transcriptions as caption to the image
        # font = ImageFont.load_default(size=16)
        # caption_height = 50 + (20 * len(transcriptions.splitlines()))
        # caption_image = Image.new("RGB", (image.width, caption_height), color="black")
        # draw = ImageDraw.Draw(caption_image)
        # draw.text((10, 10), transcriptions, fill="white", font=font)

        # # Combine the original image with the caption image
        # new_image = Image.new("RGB", (image.width, image.height + caption_height))
        # new_image.paste(image, (0, 0))
        # new_image.paste(caption_image, (0, image.height))

        # new_image.save(f"{output_path}/polygons_{filename.replace('.xml', '.jpg')}")


if __name__ == "__main__":
    get_ground_truth(
        "0001_2c8b3b76-0237-4fb8-8d4b-6b9b783b6889_label_front_0001_label.tif"
    )
    # iou, weight = compute_iou(pred_sum, gt_sum)
