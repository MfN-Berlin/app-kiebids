import os

from lxml import etree

from kiebids import config, get_logger
from kiebids.utils import extract_polygon

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
