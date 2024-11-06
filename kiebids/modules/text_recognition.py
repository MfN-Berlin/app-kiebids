import easyocr
import numpy as np
import torch
from prefect import task

from kiebids import config, get_logger, pipeline_config
from kiebids.utils import crop_image, debug_writer

module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = "" if config.mode != "debug" else f"{config['debug_path']}/{module}"
module_config = pipeline_config[module]


class TextRecognizer:
    """
    Text Recognizer class
    """

    def __init__(self):
        gpu = torch.cuda.is_available()
        self.model = easyocr.Reader([module_config.language], gpu=gpu)

    def get_text(self, image: np.array):
        # readtext() returns either an empty list if no text found or a list with only one element of text.
        # If detail=1 it would return a list of texts, but we are interested in evaluating the whole image.
        texts = self.model.readtext(
            image,
            decoder=module_config.decoder,
            text_threshold=module_config.text_threshold,
            paragraph=True,
            detail=0,
        )
        return texts[0] if texts else ""

    @task(name=module)
    @debug_writer(debug_path, module=module)
    def run(self, image: np.array, bounding_boxes: list, **kwargs):
        """
        Returns text for each bounding box in image
        Parameters:
            image: np.array
            bounding_boxes: list of bounding box coordinates of form [x_min,y_min,width,height]

        Returns:
            dictionary with bounding box and text
        """

        output = []

        for bounding_box in bounding_boxes:
            cropped_image = crop_image(image, bounding_box)

            text = self.get_text(image=cropped_image)

            output.append({"bbox": bounding_box, "text": text})

        return output
