import os
from io import BytesIO

import numpy as np
import requests
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2 as cv

from kiebids import config, get_logger
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
                gt_labels = get_ground_truth(kwargs.get("filename"))
                if gt_labels:
                    compare_layouts(bb_labels, gt_labels)

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
    xml_file = filename.replace(".jpg", ".xml")
    polygons = []

    # check if ground truth is available
    if config.evaluation_paths.hymdata and xml_file in os.listdir(config.evaluation_paths.hymdata):
        # get labels from xml file
        file_path = os.path.join(config.evaluation_paths.hymdata, xml_file)
        tree = etree.parse(file_path)
        root = tree.getroot()
        ns = {"ns": root.nsmap[None]} if None in root.nsmap else {}

        transcriptions = ""
        textlines = root.xpath("//ns:TextLine" if ns else "//TextLine", namespaces=ns)
        for textline in textlines:
            coords = textline.find("ns:Coords" if ns else "Coords", namespaces=ns)
            if coords is not None:
                polygons.append(extract_polygon(coords.get("points")))

            # unicode_elem = textline.find(".//ns:Unicode" if ns else ".//Unicode", namespaces=ns)
            # if unicode_elem is not None:
            #     transcriptions += f"{i+1}. {unicode_elem.text}\n"

    return polygons


def compare_layouts(bb_labels: list, ground_truth: list):

    # pred_masks = outputs[0]["instances"][outputs[0]["instances"].pred_classes == cat_id].pred_masks.cpu().numpy()
    # pred_sum = (pred_masks.sum(axis=0) > 0) if pred_masks.size != 0 else np.zeros((height, width)) > 0

    # gt_masks = get_target_masks(input["image_id"], cat_id)
    # gt_sum = (gt_masks.sum(axis=0) > 0) if gt_masks.size != 0 else np.zeros((height, width)) > 0

    # for now just assume 1 to 1 mapping
    for i, bb in enumerate(bb_labels):
        pred_sum = bb["segmentation"]
        gt_sum = create_polygon_mask(ground_truth[i], pred_sum.shape)

        if np.sum(pred_sum + gt_sum) == 0:
            continue

        iou, weight = compute_iou(pred_sum, gt_sum)
    # self.weights.append(weight)
    # cat_weights.append(weight)
    # cat_ious.append(iou)

    # self.update_frame_metrics(iou, gt_sum, pred_sum, cat_id)


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
    cv.fillPoly(mask, [polygon_points], 1)

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
        response = requests.get(url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))

        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image from URL: {e}")
        return None
    except OSError as e:
        logger.error(f"Error opening image: {e}")
        return None


def process_xml_files(folder_path, output_path):
    """
    Process XML files in the given folder path and
    save the images with polygons and transcriptions in the output path.
    """

    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
    for filename in tqdm(files[:10], desc="Processing XML files"):
        file_path = os.path.join(folder_path, filename)
        tree = etree.parse(file_path)
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
            grayscale_image = image.convert("L")

            kernel = np.ones((5, 5), np.uint8)
            # Convert the grayscale PIL image to a NumPy array (of type uint8)
            grayscale_np = np.array(grayscale_image, dtype=np.uint8)

            cv.imwrite(f"{output_path}/{filename.replace('.xml', '_gs.jpg')}", grayscale_np)

            # Apply adaptive thresholding using cv.adaptiveThreshold
            thresholded_image = cv.adaptiveThreshold(
                grayscale_np, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 17, 2
            )

            ret, binary = cv.threshold(grayscale_np, 100, 255, cv.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(thresholded_image, cv.MORPH_OPEN, kernel)
            closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

            up_points = (grayscale_np.shape[0] * 4, grayscale_np.shape[1] * 4)
            resized_up = cv.resize(binary, up_points, interpolation=cv.INTER_LINEAR)

            image.save(f"{output_path}/{filename.replace('.xml', '_orig.jpg')}")
            cv.imwrite(f"{output_path}/{filename.replace('.xml', '_bw.jpg')}", thresholded_image)
            cv.imwrite(f"{output_path}/{filename.replace('.xml', '_open.jpg')}", opening)
            cv.imwrite(f"{output_path}/{filename.replace('.xml', '_closing.jpg')}", closing)
            cv.imwrite(f"{output_path}/{filename.replace('.xml', '_binary.jpg')}", binary)

        # lookup for polygon coordinates and transcriptions
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
    get_ground_truth("66fbd9dc-e75c-46f5-8072-af3a10865de4_label_front_0002_label.jpg")
    # iou, weight = compute_iou(pred_sum, gt_sum)
