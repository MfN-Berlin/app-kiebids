import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import cv2
from prefect import task

from kiebids import config
from kiebids.utils import bounding_box_to_coordinates


def create_page_content(
    filename: str,
    tr_result: list,
    st_result: list,
    el_result: list,
    width: int = None,
    height: int = None,
):
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
    ET.SubElement(metadata, "Creator").text = config.creator_info
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    ET.SubElement(metadata, "LastChange").text = datetime.now().isoformat()

    page = ET.SubElement(root, "Page")
    page.set("imageFilename", filename)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))

    reading_order = ET.SubElement(page, "ReadingOrder")
    ordered_group = ET.SubElement(reading_order, "OrderedGroup")
    ordered_group.set("id", "ro_group_1")

    # Create RegionRefIndexed for each TextRegion
    for idx, _ in enumerate(tr_result):
        region_ref = ET.SubElement(ordered_group, "RegionRefIndexed")
        region_ref.set("index", str(idx))
        region_ref.set("regionRef", f"TextRegion_{idx}")

    # Create TextRegion for each bbox-text pair
    for idx, label in enumerate(tr_result):
        coords = bounding_box_to_coordinates(label["bbox"])
        text = label["text"]

        # TextRegion
        text_region = ET.SubElement(page, "TextRegion")
        text_region.set("id", f"TextRegion_{idx}")
        text_region.set("custom", f"readingOrder {{index:{idx};}}")

        coords_elem = ET.SubElement(text_region, "Coords")
        coords_elem.set("points", coords)

        # Tagging per region
        tags = st_result[idx]

        # Textline - currently only one per TextRegion
        text_line = ET.SubElement(text_region, "TextLine")
        text_line.set("id", "line_1")

        # Make custom string for the current TextLine
        custom_string = f"readingOrder {{index:{idx};}}"
        for tag in tags:
            custom_string += f" {tag.label_} {{offset:{tag.start_char}; length:{tag.end_char - tag.start_char};}}"

        text_line.set("custom", custom_string)
        coords_elem_line = ET.SubElement(text_line, "Coords")
        coords_elem_line.set("points", coords)
        baseline_elem_line = ET.SubElement(text_line, "Baseline")
        baseline_elem_line.set("points", coords)
        text_equiv_line = ET.SubElement(text_line, "TextEquiv")
        ET.SubElement(text_equiv_line, "Unicode").text = text

        # Add TextEquiv for the whole region
        text_equiv = ET.SubElement(text_region, "TextEquiv")
        ET.SubElement(text_equiv, "Unicode").text = text

        if f"region_{idx}" in el_result:
            for i, le in enumerate(el_result[f"region_{idx}"]):
                entity_linking = ET.SubElement(text_region, "EntityLinking")
                entity_linking.set("id", f"entity_linking_{i}")
                entity_linking.set("text", str(le["span"]))
                entity_linking.set("label", le["span"].label_)
                entity_linking.set("geoname_ids", le["geoname_ids"])

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
def write_page_xml(current_image_name, tr_result, st_result, el_result):
    """
    Writes the PAGE XML file for the given image.

    """
    image_path = Path(config.image_path) / current_image_name
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    root = create_page_content(
        filename=current_image_name,
        tr_result=tr_result,
        st_result=st_result,
        el_result=el_result,
        width=width,
        height=height,
    )

    output_path = (
        Path(config.output_path) / f"{os.path.splitext(current_image_name)[0]}.xml"
    )
    save_xml(root, output_path)
