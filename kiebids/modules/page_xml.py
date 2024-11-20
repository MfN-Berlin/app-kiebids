import os

from datetime import datetime
from pathlib import Path
from prefect import task
import xml.etree.ElementTree as ET


def create_page_content(filename: str, bbox_text: list[dict]):
    """Create a PAGE XML file structure with multiple TextRegions."""
    nsmap = {
        None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    root = ET.Element(
        f"{{{nsmap[None]}}}PcGts",
        {
            f"{{{nsmap['xsi']}}}schemaLocation": f"{nsmap[None]} {nsmap[None]}/pagecontent.xsd"
        },
    )

    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Your Creator Info"
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    ET.SubElement(metadata, "LastChange").text = datetime.now().isoformat()

    page = ET.SubElement(root, "Page")
    page.set("imageFilename", filename)
    page.set("imageWidth", "6720")
    page.set("imageHeight", "4480")

    reading_order = ET.SubElement(page, "ReadingOrder")
    ordered_group = ET.SubElement(reading_order, "OrderedGroup")
    ordered_group.set("id", "ro_group_1")

    # Create RegionRefIndexed for each TextRegion
    for idx, _ in enumerate(bbox_text):
        region_ref = ET.SubElement(ordered_group, "RegionRefIndexed")
        region_ref.set("index", str(idx))
        region_ref.set("regionRef", f"TextRegion_{idx}")

    # Create TextRegion for each bbox-text pair
    for idx, label in enumerate(bbox_text):
        coords = " ".join(str(x) for x in label["bbox"])
        text = label["text"]

        text_region = ET.SubElement(page, "TextRegion")
        text_region.set("id", f"TextRegion_{idx}")
        text_region.set("custom", f"readingOrder {{index:{idx};}}")

        coords_elem = ET.SubElement(text_region, "Coords")
        coords_elem.set("points", coords)

        text_line = ET.SubElement(text_region, "TextLine")
        text_line.set("id", "line_1")

        text_equiv = ET.SubElement(text_line, "TextEquiv")
        ET.SubElement(text_equiv, "Unicode").text = text

    return root


def save_xml(root, output_path):
    """Save the XML tree to a file with indentation."""
    tree = ET.ElementTree(root)

    ET.register_namespace(
        "", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    )
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    def indent(elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                indent(subelem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            # Adjust the last child's tail to align closing tag
            if len(elem) > 0:
                elem[-1].tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(root)

    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


@task
def write_page_xml(output_path, filename, result):
    """
    Writes the PAGE XML file for the given image.
    """
    output_path = Path(output_path) / f"{os.path.splitext(filename)[0]}.xml"
    root = create_page_content(filename, result)
    save_xml(root, output_path)
