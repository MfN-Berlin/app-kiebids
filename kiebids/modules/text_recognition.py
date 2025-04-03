import easyocr
import numpy as np
import torch
from PIL import Image
from prefect import task
from transformers import AutoModelForCausalLM

from kiebids import config, get_logger, pipeline_config, run_id
from kiebids.modules.evaluation import evaluator
from kiebids.utils import crop_image, debug_writer

module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = (
    "" if config.mode != "debug" else f"{config['debug_path']}/{module}/{run_id}"
)
module_config = pipeline_config[module]


class TextRecognizer:
    """
    Text Recognizer class
    """

    def __init__(self):
        logger.info(f"Loading text recognition model ({module_config.model})...")

        MODEL_REGISTRY = {"easyocr": EasyOcr, "moondream": Moondream}

        if module_config.model in MODEL_REGISTRY:
            self.model = MODEL_REGISTRY[module_config.model]()
        else:
            logger.warning(
                "Model name '%s' found in the workflow_config.yaml is not part of the available models: %s. Falling back to default model easyocr.",
                module_config.model,
                list(MODEL_REGISTRY.keys()),
            )
            self.model = EasyOcr()

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

            text = self.model.get_text(image=cropped_image)

            output.append({"bbox": bounding_box, "text": text})

        return output


class EasyOcr:
    """
    EasyOcr
    """

    def __init__(self):
        self.model = easyocr.Reader(
            [module_config.easyocr.language], gpu=torch.cuda.is_available()
        )

    def get_text(self, image: np.array):
        """
        Returns text from image.
        """
        texts = self.model.readtext(
            image,
            decoder=module_config.easyocr.decoder,
            text_threshold=module_config.easyocr.text_threshold,
            paragraph=module_config.easyocr.paragraph,
            detail=0,
            y_ths=module_config.easyocr.y_ths,
        )
        return "\n".join(texts) if texts else ""


class Moondream:
    """
    Moondream 1.9B 2025-01-09 Release
    Huggingface: https://huggingface.co/vikhyatk/moondream2
    Documentation: https://docs.moondream.ai/
    Blog post: https://moondream.ai/blog/introducing-a-new-moondream-1-9b-and-gpu-support
    """

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            module_config.moondream.name,
            revision=module_config.moondream.revision,
            trust_remote_code=module_config.moondream.trust_remote_code,
            device_map={"": "cuda"} if torch.cuda.is_available() else None,
        )
        self.prompt = module_config.moondream.prompt

    def get_text(self, image: np.array):
        pil_image = Image.fromarray(image)
        text = self.model.query(pil_image, self.prompt).get("answer", "")
        return self.clean_text(text)

    def clean_text(self, text):
        """
        Moondream specific text cleaning.
        """
        return text.replace("\n\n", "\n").strip()
