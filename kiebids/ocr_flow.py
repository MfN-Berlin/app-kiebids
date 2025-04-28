import argparse
import os

import fiftyone as fo
from prefect import flow
from tqdm import tqdm

from kiebids import config, evaluation_writer, fiftyone_dataset, pipeline_config, run_id
from kiebids.modules.entity_linking import EntityLinking
from kiebids.modules.layout_analysis import LayoutAnalyzer
from kiebids.modules.page_xml import write_page_xml
from kiebids.modules.preprocessing import preprocessing
from kiebids.modules.semantic_tagging import SemanticTagging
from kiebids.modules.text_recognition import TextRecognizer
from kiebids.utils import get_kiebids_logger

pipeline_name = pipeline_config.pipeline_name


# init objects/models for every stage
layout_analyzer = LayoutAnalyzer()
text_recognizer = TextRecognizer()
semantic_tagging = SemanticTagging()
entity_linking = EntityLinking()


@flow(name=pipeline_name, log_prints=True)
def ocr_flow(max_images: int, image_path: str = config.image_path):
    logger = get_kiebids_logger("kiebids_flow")
    logger.info("Starting app-kiebids... Run ID: %s", run_id)
    logger.info("Image path: %s", image_path)

    # resetting config image path.
    # TODO this should be done in a better way
    config.image_path = image_path

    # reinit evaluation with every main flow run
    if evaluation_writer:
        evaluation_writer.init_metrics()

    images_list = sorted(os.listdir(image_path))
    max_images = max_images if max_images > 0 else len(images_list)
    # Process images sequentially
    for image_index, filename in enumerate(tqdm(images_list[:max_images])):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
            continue

        try:
            single_image_flow(
                filename=filename,
                image_index=image_index,
            )
        except Exception as e:
            logger.error("Error processing image %s: %s", filename, str(e))
            continue

        # write evaluation tables only at certain intervals
        if (
            evaluation_writer
            and (image_index + 1) % config.evaluation.summary_interval == 0
        ):
            evaluation_writer.create_tables()

    # final writing
    if evaluation_writer:
        evaluation_writer.create_tables()


@flow(flow_run_name="{filename}")
def single_image_flow(filename, image_index):
    # accepts image path. outputs image
    preprocessed_image = preprocessing(current_image_name=filename)

    # accepts image. outputs image and bounding boxes. if debug the write snippets to disk
    la_result = layout_analyzer.run(
        image=preprocessed_image,
        current_image_name=filename,
        current_image_index=image_index,
    )

    # accepts image and bounding boxes. returns. if debug the write snippets with corresponding text to disk
    tr_result = text_recognizer.run(  # noqa: F841
        image=preprocessed_image,
        bounding_boxes=[bb_label["bbox"] for bb_label in la_result],
        current_image_name=filename,
        current_image_index=image_index,
    )

    # only have gt for single exhibit labels (regions). in cases when multiple labels are present, we need a way to map gt region to prediction region at hand
    st_result = semantic_tagging.run(  # noqa: F841
        texts=[result["text"] for result in tr_result],
        current_image_name=filename,
        current_image_index=image_index,
    )

    el_result = entity_linking.run(
        st_result=st_result,
        current_image_name=filename,
    )

    if evaluation_writer:
        evaluation_writer.create_table()

    # write results to PAGE XML
    write_page_xml(
        current_image_name=filename,
        tr_result=tr_result,
        st_result=st_result,
        el_result=el_result,
    )


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
                parameters={
                    "max_images": config.max_images,
                    "image_path": config.image_path,
                },
            )
        else:
            ocr_flow(max_images=config.max_images)

    if not config.disable_fiftyone:
        fiftyone_session = fo.launch_app(fiftyone_dataset)
        fiftyone_session.wait()
