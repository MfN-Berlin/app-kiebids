from lxml import etree
import os
from kiebids import get_logger

logger = get_logger(__name__)


def read_xml(file_path: str) -> dict:
    """
    Parses an XML file and extracts information about pages, text regions, and text lines.
    Args:
        file_path (str): The path to the XML file to be parsed.
    Returns:
        dict: A dictionary containing the extracted information with the following structure:
            {
                "image_filename": str,  # The filename of the image associated with the page
                "image_width": str,     # The width of the image
                "image_height": str,    # The height of the image
                "text_regions": [       # A list of text regions
                    {
                        "id": str,           # The ID of the text region
                        "orientation": str,  # The orientation of the text region
                        "coords": str,       # The coordinates of the text region
                        "text": str,         # The text content of the whole text region
                        "text_lines": [      # A list of text lines within the text region
                            {
                                "id": str,        # The ID of the text line
                                "coords": str,    # The coordinates of the text line
                                "baseline": str,  # The baseline coordinates of the text line
                                "text": str       # The text content of the text line
                            }
                        ]
                    }
                ]
            }
    """

    tree = etree.parse(file_path)  # noqa: S320  # Using `lxml` to parse untrusted data is known to be vulnerable to XML attacks
    root = tree.getroot()

    ns = {"ns": root.nsmap[None]} if None in root.nsmap else {}

    output = {}

    page = root.find(".//ns:Page", namespaces=ns)

    output["image_filename"] = page.get("imageFilename")
    output["image_width"] = page.get("imageWidth")
    output["image_height"] = page.get("imageHeight")

    text_regions = page.findall(".//ns:TextRegion", namespaces=ns)

    text_regions_list = []

    for text_region in text_regions:
        text_region_dict = {}
        text_region_dict["id"] = text_region.get("id")
        text_region_dict["orientation"] = text_region.get("orientation")
        text_region_dict["coords"] = text_region.find(
            ".//ns:Coords", namespaces=ns
        ).get("points")

        text = text_region.findall(".//ns:TextEquiv", namespaces=ns)[-1]

        text = (
            text.find(".//ns:Unicode", namespaces=ns).text if text is not None else ""
        )
        text_region_dict["text"] = text

        text_lines = text_region.findall(".//ns:TextLine", namespaces=ns)
        text_line_list = []

        for text_line in text_lines:
            text_line_dict = {}
            text_line_dict["id"] = text_line.get("id")
            text_line_dict["coords"] = text_line.find(
                ".//ns:Coords", namespaces=ns
            ).get("points")
            text_line_dict["baseline"] = text_line.find(
                ".//ns:Baseline", namespaces=ns
            ).get("points")
            text = text_line.find(".//ns:TextEquiv", namespaces=ns)
            text = (
                text.find(".//ns:Unicode", namespaces=ns).text
                if text is not None
                else ""
            )
            text_line_dict["text"] = text
            text_line_list.append(text_line_dict)

        text_region_dict["text_lines"] = text_line_list

        text_regions_list.append(text_region_dict)

        output["text_regions"] = text_regions_list

    return output


def get_ground_truth_text(filename: str, xml_path: str):
    """
    params:
    filename: name of file
    xml_path: path to folder with xml_files
    """
    file_path = os.path.join(xml_path, filename + ".xml")

    # get the xml file
    if os.path.exists(file_path):
        ground_truth_data = read_xml(file_path)
    else:
        logger.info("ground truth file not found at: %s ", {xml_path})
        return None

    # Return the text content of the text regions
    text_result = [region["text"] for region in ground_truth_data["text_regions"]]

    return text_result
