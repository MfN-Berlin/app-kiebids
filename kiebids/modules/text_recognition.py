import easyocr
import numpy as np
import torch
from PIL import Image
from prefect import task
from transformers import AutoModelForCausalLM

from kiebids import config, get_logger, pipeline_config
from kiebids.modules.evaluation import evaluator
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
    @evaluator(module=module)
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


class Moondream(TextRecognizer):
    """
    Moondream 1.9B 2025-01-09 Release
    Huggingface: https://huggingface.co/vikhyatk/moondream2
    Documentation: https://docs.moondream.ai/
    Blog post: https://moondream.ai/blog/introducing-a-new-moondream-1-9b-and-gpu-support
    """

    def __init__(self):
        gpu = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": "cuda"} if gpu else None,
        )
        self.prompt = """
            Transcribe all printed and handwritten text on this label of a specimen
            from a collection of a museum for natural history, being especially
            careful to preserve any scientific names, dates, and location information.
            Maintain the original formatting and line breaks. Most text is in German.
            """

    def get_text(self, image: np.array):
        pil_image = Image.fromarray(image)
        return self.model.query(pil_image, self.prompt)["answer"]
