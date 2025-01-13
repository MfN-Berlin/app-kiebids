import cv2
import numpy as np
import torch
from prefect import task
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from kiebids import config, get_logger, pipeline_config
from kiebids.modules.evaluation import evaluator
from kiebids.utils import debug_writer

module = __name__.split(".")[-1]
logger = get_logger(module)
logger.setLevel(config.log_level)

debug_path = "" if config.mode != "debug" else f"{config['debug_path']}/{module}"
module_config = pipeline_config[module]


class LayoutAnalyzer:
    def __init__(self):
        model_path = module_config["model_path"]
        self.mask_generator = self.load_model(model_path)

    @debug_writer(debug_path, module=module)
    @evaluator(module=module)
    @task(name=module)
    def run(self, image, **kwargs):  # pylint: disable=unused-argument
        masks = self.mask_generator.generate(image)
        for mask in masks:
            # ensure bbox coords are ints; leider the model sometimes returns floats
            bbox = [int(coord) for coord in mask["bbox"]]
            mask["bbox"] = bbox
            height, width = image.shape[:2]
            mask["normalized_bbox"] = [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height,
            ]

        label_masks = self.filter_masks(masks)

        return label_masks

    def load_model(self, model_path):
        logger.info(f"Loading segment anything model from {model_path} ...")
        sam = sam_model_registry[module_config["model_type"]](checkpoint=model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=module_config["points_per_side"],
            pred_iou_thresh=module_config["pred_iou_thresh"],
            stability_score_thresh=module_config["stability_score_thresh"],
            crop_n_layers=module_config["crop_n_layers"],
            min_mask_region_area=module_config["min_mask_region_area"],
            output_mode=module_config["output_mode"],  # "uncompressed_rle"
        )
        return mask_generator

    def filter_masks(self, masks):
        """Sort masks by area in descending order and keep only those that mask labels :)"""
        # If there is only one mask, return it
        if len(masks) == 1:
            return masks

        kernel = np.ones(
            (module_config.closing_kernel, module_config.closing_kernel), np.uint8
        )
        sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

        # Keep only masks that cover more than 1% of the image
        label_masks = []
        total_area = sorted_masks[0]["segmentation"].size
        for mask in sorted_masks:
            area = mask["area"]

            # Filter by areas that cover more than 1% of the image
            if (area / total_area) > 0.01:
                [x, y, w, h] = mask["bbox"]
                bbox_area = w * h

                binary_mask = np.array(mask["segmentation"].copy() * 1, dtype=np.uint8)

                # Closing little holes in mask
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

                contours, hierarchy = cv2.findContours(
                    binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                holes_present = any(
                    hierarchy[0][i][3] != -1 for i in range(len(contours))
                )

                # excluding masks with holes and masks with 10% differences to bb
                if (area / bbox_area) > 0.9 and not holes_present:
                    label_masks.append(mask)

        if label_masks == []:
            # If no masks are found, return the largest mask
            return [sorted_masks[0]]

        return label_masks
