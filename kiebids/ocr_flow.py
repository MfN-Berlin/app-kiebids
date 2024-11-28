import argparse
import os

import fiftyone as fo

# commented out for now to avoid tensorflow loading
# from modules.semantic_labeling import semantic_labeling
from prefect import flow
from tqdm import tqdm

from kiebids import config, fiftyone_dataset, get_logger, pipeline_config
from kiebids.modules.layout_analysis import LayoutAnalyzer
from kiebids.modules.page_xml import write_page_xml
from kiebids.modules.preprocessing import preprocessing
from kiebids.modules.text_recognition import TextRecognizer

pipeline_name = pipeline_config.pipeline_name
logger = get_logger(pipeline_name)


@flow(name=pipeline_name, log_prints=True)
def ocr_flow():
    os.makedirs(config.output_path, exist_ok=True)

    # init objects/models for every stage
    logger.info("Loading Layout analysis Model...")
    layout_analyzer = LayoutAnalyzer()
    logger.info("Loading Text recognition Model...")
    text_recognizer = TextRecognizer()

    # Process images sequentially
    for image_index, filename in enumerate(
        tqdm(os.listdir(config.image_path)[: config.max_images])
    ):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
            continue

        logger.info("Processing image %s from %s.", filename, config.image_path)

        # accepts image path. outputs image
        preprocessed_image = preprocessing(current_image_name=filename)

        # accepts image. outputs image and bounding boxes. if debug the write snippets to disk
        bb_labels = layout_analyzer.run(
            image=preprocessed_image,
            current_image_name=filename,
            current_image_index=image_index,
        )
        # accepts image and bounding boxes. returns. if debug the write snippets with corresponding text to disk
        results = text_recognizer.run(  # noqa: F841
            image=preprocessed_image,
            bounding_boxes=[bb_label["bbox"] for bb_label in bb_labels],
            current_image_name=filename,
            current_image_index=image_index,
        )

        # semantic_labeling.run
        # semantic_labeling_output_dir = semantic_labeling(layout_analysis_output_dir, config.output_path)

        # entity_linking.run
        # entity_linking(image_path, config.output_path)

        # write results to PAGE XML
        write_page_xml(config.output_path, filename, results)

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
    parser.add_argument(
        "--fiftyone-only",
        action="store_true",
        help="launches only the fo app. Assuming available datasets.",
    )

    args = parser.parse_args()

    if not args.fiftyone_only:
        if args.serve_deployment:
            ocr_flow.serve(
                name=pipeline_config.deployment_name,
                parameters={},
            )
        else:
            ocr_flow()

    if config.mode == "debug":
        fiftyone_session = fo.launch_app(fiftyone_dataset)
        fiftyone_session.wait()
