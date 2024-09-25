import easyocr
import torch

import numpy as np

from prefect import task

from kiebids import config, pipeline_config, get_logger

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
        self.language = module_config["language"]
        self.text_threshold = module_config["text_threshold"]
        self.decoder = module_config["decoder"]

        self.model = easyocr.Reader([self.language], gpu=gpu)

    def get_text(self, image: np.array):
        text = self.model.readtext(
            image, decoder=self.decoder, text_threshold=self.text_threshold, paragraph=True, detail=0
        )
        if len(text) == 0:
            return ""
        else:
            return text[0]

    def crop_image(self, image: np.array, bounding_box: list[int]):
        """get the cropped image from bounding boxes"""
        x, y, w, h = bounding_box
        return image[y : y + h, x : x + w]

    @task
    def run(self, image: np.array, bounding_boxes: list):
        """
        Returns text for each bounding box in image
        Parameters:
            image: np.array
            bounding_boxes: list of bounding box coordinates of form [x,y,w,h].

        Returns:
            dictionary with bounding box and text
        """

        output = []

        for bounding_box in bounding_boxes:
            cropped_image = self.crop_image(image, bounding_box)

            text = self.get_text(image=cropped_image)

            output.append({"bbox": bounding_box, "text": text})

        return output
