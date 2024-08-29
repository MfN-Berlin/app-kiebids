from lxml import etree
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageDraw
import requests

def draw_bounding_box(
    frame: np.ndarray, scores: list[float], boxes: list[list[float]], labels: list[str]
) -> np.ndarray:
    frame_bb = np.copy(frame)
    height, width = frame_bb.shape[:2]
    valid_boxes = []

    for n, box in enumerate(boxes):
        x_center, y_center, width_norm, height_norm = box
        w1 = int((x_center - width_norm / 2) * width)
        h1 = int((y_center - height_norm / 2) * height)
        w2 = int((x_center + width_norm / 2) * width)
        h2 = int((y_center + height_norm / 2) * height)

        # Check if the bounding box is too large (we don't want frame-sized boxes)
        if (w2 - w1) / width < 0.95 and (h2 - h1) / height < 0.95:
            valid_boxes.append((w1, h1, w2, h2, labels[n], scores[n]))

    for w1, h1, w2, h2, label, score in valid_boxes:
        cv2.rectangle(frame_bb, (w1, h1), (w2, h2), (0, 255, 0), 2)
        cv2.putText(frame_bb, f"{label}: {score:.2f}", (w1 + 8, h1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame_bb if valid_boxes else None


def load_image_from_url(url):
    try:
        # Fetch the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        # Load the image into memory
        image = Image.open(BytesIO(response.content))

        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return None
    except IOError as e:
        print(f"Error opening image: {e}")
        return None


def draw_polygon_on_image(image_path, coordinates, output_path):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Convert the string of coordinates into a list of tuples
    points = [tuple(map(int, point.split(','))) for point in coordinates.split()]

    # Draw the polygon
    draw.polygon(points, outline="red", fill=None)

    # Save the image with the drawn polygon
    image.save(output_path)
    print(f"Image saved to {output_path}")




if __name__ == "__main__":
    # load and save image
    # image = load_image_from_url("https://files.transkribus.eu/Get?id=QFUJGCEQCZMEPSENVPIRJGQA")
    # image.save("./data/test_image.jpg")

    # access points in xml
    tree = etree.parse('data/test_transcribus.xml')
    root = tree.getroot()
    ns = {'ns': root.nsmap[None]} if None in root.nsmap else {}

    # XPath query to find all TextLine elements, considering namespaces if they exist
    textlines = root.xpath('//ns:TextLine' if ns else '//TextLine', namespaces=ns)

    # Iterate over each TextLine and extract the points from Coords
    for textline in textlines:
        coords = textline.find('ns:Coords' if ns else 'Coords', namespaces=ns)
        if coords is not None:
            points = coords.get('points')
            print(points)


    # draw polygon on image
    # image_path = "./data/test_image.jpg"
    # output_path = "./data/pol_image.jpg"
    # coordinates = "95,101 140,108 178,99 192,108 227,108 280,94 302,103 337,95 360,101 360,51 333,36 309,40 285,32 248,47 203,30 149,42 112,27 96,34"
    # draw_polygon_on_image(image_path, coordinates, output_path)


# # Example usage
# image_url = "https://example.com/path/to/your/image.jpg"
# image = load_image_from_url(image_url)
# if image:
#     print("Image loaded successfully.")


# tree = etree.parse('data/test_transcribus.xml')
# root = tree.getroot()

# # Access elements in the XML file
# for element in root.iter():
#     # TODO read image from given imgUrl
#     print(element.tag, element.attrib)

# # examples
# for text_region in root.findall(".//TextRegion"):
#     print(text_region.attrib)

# for text_line in root.findall(".//TextLine"):
#     line_text = "".join(text_line.itertext())
#     print(line_text)

# for element in root.findall(".//TextRegion"):
#     print(element.get('id'))  # Example attribute
