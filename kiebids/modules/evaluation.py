import os
from io import BytesIO

import requests
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from kiebids.utils import draw_polygon_on_image
from kiebids import config, get_logger, current_dataset

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
    for filename in tqdm(files, desc="Processing XML files"):
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
