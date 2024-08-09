import os
import yaml
import cv2

import numpy as np
from pathlib import Path
from prefect import task

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Import config 
script_path = Path(__file__).parent.parent.resolve()
with open(script_path / "ocr_config.yaml") as file:
    pipeline_config = yaml.safe_load(file)

layout_analysis_config = pipeline_config["layout_analysis"]

# Create debugging path
DEBUG_PATH = script_path.parent / "data" / "debug" / "layout_analysis"
os.makedirs(DEBUG_PATH, exist_ok=True)

@task
def layout_analysis(image_path, output_path, debug=False):
    # TODO: Put the paths in a config file somewhere? 
    OUTPUT_DIR_LAYOUT_ANALYSIS = Path(output_path) / "layout_analysis"
    os.makedirs(OUTPUT_DIR_LAYOUT_ANALYSIS, exist_ok=True)
    # TODO: Rewrite the code so that the model doesn't have to be loaded new for each image
    # TODO: No hardcoding - put in config file somwhere? 
    model_path = script_path.parent / "models" / layout_analysis_config["name"]
    mask_generator = load_model(model_path)

    image_orig = cv2.imread(image_path)
    masks = mask_generator.generate(image_orig)
    label_masks = filter_masks(masks)

    if debug==True:
        save_anns(label_masks, DEBUG_PATH)

    image_name = image_path.split('/')[-1].split('.')[0]
    plot_and_save_bbox_images(image_orig, label_masks, image_name, OUTPUT_DIR_LAYOUT_ANALYSIS)
    return OUTPUT_DIR_LAYOUT_ANALYSIS


def load_model(model_path): 

    # device = "cpu"
    sam = sam_model_registry[layout_analysis_config['model_type']](checkpoint=model_path) 

    # TODO: Change to logging 
    print(f"Loading segment anything Model")
    mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=layout_analysis_config['points_per_side'],
            pred_iou_thresh=layout_analysis_config['pred_iou_thresh'],
            stability_score_thresh=layout_analysis_config['stability_score_thresh'],
            crop_n_layers=layout_analysis_config['crop_n_layers'],
            min_mask_region_area=layout_analysis_config['min_mask_region_area'],
            output_mode=layout_analysis_config["output_mode"]  # "uncompressed_rle"
        )
    return mask_generator   


def save_anns(anns, output_path):
    if len(anns) == 0:
        print("No annotations found.")
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_mask = np.array([1.0, 0.0, 1.0, 0.35])  # Create a pink color mask with some transparency
        img[m] = color_mask
    # ax.plot(img) 
    print("Saving annotation analysis to %s", output_path)
    #ax.figure.savefig(os.path.join(output_path, "annotation_analysis.png"))
    cv2.imwrite(os.path.join(output_path, "annotation_analysis_cv.png"), img)
    # plot_and_save_masked_images(image_orig, selected_masks, image_path)

def plot_and_save_bbox_images(image, masks, image_name, output_dir):
    """
    Plot and save individual images for each mask, using the bounding box to crop the image.
    
    Args:
    image (numpy.ndarray): The original image as a numpy array (height, width, 3).
    masks (list): A list of dictionaries, each containing a 'bbox' key with [x, y, width, height].
    output_dir (str): Directory to save the output images.
    """

    for i, mask in enumerate(masks, 1):
        bbox = mask['bbox']
        x, y, w, h = bbox
        
        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]
        
        # Save the cropped image
        output_path = os.path.join(output_dir, f'{image_name}_{i}.png')
        cv2.imwrite(output_path, cropped_image)
        
        print(f"Saved bounding box image to {output_path}")

def filter_masks(masks):
    """Sort masks by area in descending order and keep only those that mask labels :)"""
    # If there is only one mask, return it
    if len(masks) == 1:
        return masks

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # Keep only masks that cover more than 1% of the image
    label_masks = []
    total_area = sorted_masks[0]['segmentation'].size
    for mask in sorted_masks:
        area = mask['area']
        # Filter by areas that cover more than 1% of the image
        if (area / total_area) > 0.01:
            # print(f"Area percentage: {(area / total_area)}")
            # Filter by areas where the segmentation mask covers most of the bbox area
            [x, y, w, h] = mask["bbox"]
            bbox_area = w * h
            # print(f"seg to bbox area ratio: {(area / bbox_area)}")
            if (area / bbox_area) > 0.9:
                label_masks.append(mask)

    if label_masks == []:
        # If no masks are found, return the largest mask
        return [sorted_masks[0]]

    return label_masks
