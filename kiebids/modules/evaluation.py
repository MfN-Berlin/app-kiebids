import os
from io import BytesIO

import requests
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from kiebids import config, getLogger

logger = getLogger(__name__, log_level=config.log_level)


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


if __name__ == "__main__":
    # access points in xml
    process_xml_files(
        os.path.join(
            config.shared_folder,
            "hymdata_sample/20230511T160908__coll.mfn-berlin.de_u_78a081",
        ),
        os.path.join(config.shared_folder, "hymdata_overlayed"),
    )
