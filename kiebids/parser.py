from lxml import etree


def parse_xml(file_path: str) -> dict:
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
    ns = {"ns": tree.getroot().nsmap.get(None, "")}

    page = tree.find(".//ns:Page", namespaces=ns)
    output = {
        "image_filename": page.get("imageFilename"),
        "image_width": page.get("imageWidth"),
        "image_height": page.get("imageHeight"),
        "text_regions": [],
    }

    for region in page.findall(".//ns:TextRegion", namespaces=ns):
        text_region = {
            "id": region.get("id"),
            "orientation": region.get("orientation"),
            "coords": region.find(".//ns:Coords", namespaces=ns).get("points"),
            "text": (
                region.findall(".//ns:TextEquiv", namespaces=ns)[-1]
                .find(".//ns:Unicode", namespaces=ns)
                .text
                or ""
            ),
            "text_lines": [],
        }

        for line in region.findall(".//ns:TextLine", namespaces=ns):
            text_region["text_lines"].append(
                {
                    "id": line.get("id"),
                    "coords": line.find(".//ns:Coords", namespaces=ns).get("points"),
                    "baseline": line.find(".//ns:Baseline", namespaces=ns).get(
                        "points"
                    ),
                    "text": (
                        line.find(".//ns:TextEquiv", namespaces=ns)
                        .find(".//ns:Unicode", namespaces=ns)
                        .text
                        or ""
                    ),
                }
            )

        output["text_regions"].append(text_region)

    return output
