import os
from tqdm import tqdm
from pathlib import Path
import argparse

from kiebids.modules.layout_analysis import LayoutAnalyzer
from kiebids.modules.preprocessing import preprocessing
from kiebids.modules.text_recognition import TextRecognizer
from kiebids import config, pipeline_config, get_logger

# commented out for now to avoid tensorflow loading
# from modules.semantic_labeling import semantic_labeling
from prefect import flow

pipeline_name = pipeline_config.pipeline_name
logger = get_logger(pipeline_name)


@flow(name=pipeline_name, log_prints=True, retries=3)
def ocr_flow():

    # init objects/models for every stage
    layout_analyzer = LayoutAnalyzer()
    text_recognizer = TextRecognizer()

    # Process images sequentially
    for filename in tqdm(os.listdir(config.image_path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
            continue

        logger.info("Processing image %s from %s.", filename, config.image_path)

        # accepts image path. outputs image
        preprocessed_image = preprocessing(image_path=Path(config.image_path) / filename)

        # accepts image. outputs image and bounding boxes. if debug the write snippets to disk
        bb_labels = layout_analyzer.run(image=preprocessed_image, filename=filename)

        # accepts image and bounding boxes. returns. if debug the write snippets with corresponding text to disk
        recognized_text = text_recognizer.run(
            image=preprocessed_image, bounding_boxes=[bb_label["bbox"] for bb_label in bb_labels], filename=filename
        )

        # semantic_labeling.run
        # semantic_labeling_output_dir = semantic_labeling(layout_analysis_output_dir, output_path)

        # entity_linking.run
        # entity_linking(image_path, output_path)

        # write results to PAGE Format

    # # Process images concurrently
    # futures = process_single_image.map(image_paths, OUTPUT_DIR)

    # # Wait for all futures to complete and gather results
    # results = [future.result() for future in futures]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serve-deployment",
        action="store_true",
        help="activate deployment serving mode",
    )

    args = parser.parse_args()
    if args.serve_deployment:
        ocr_flow.serve(
            name=pipeline_config.deployment_name,
            parameters={},
        )
    else:
        ocr_flow()
