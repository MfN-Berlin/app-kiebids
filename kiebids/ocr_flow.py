import os
import sys

from pathlib import Path
from prefect import flow, task

from modules.preprocessing import preprocessing
from modules.layout_analysis import layout_analysis
from modules.text_recognition import text_recognition
from modules.semantic_labeling import semantic_labeling
from modules.entity_linking import entity_linking


BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = Path(os.environ.get("INPUT_DIR", BASE_DIR / "data" / "input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "output"))


@task
def process_single_image(image_path, output_path, debug=True):
    preprocessing_output_path = preprocessing(image_path, output_path, debug=debug)
    layout_analysis_output_dir = layout_analysis(preprocessing_output_path, output_path, debug=debug)
    text_recognition_output_dir = text_recognition(layout_analysis_output_dir, output_path, debug=debug)
    semantic_labeling_output_dir = semantic_labeling(layout_analysis_output_dir, output_path, debug=debug)
    # entity_linking(image_path, output_path, debug=debug)


@flow(name="KIEBIDS pipeline", log_prints=True)
def ocr_flow():

    image_paths = [
        os.path.join(INPUT_DIR, file)
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif"))
    ]

    # Process images sequentially
    for image_path in image_paths:
        process_single_image(image_path, str(OUTPUT_DIR), debug=True)

    # # Process images concurrently
    # futures = process_single_image.map(image_paths, OUTPUT_DIR)

    # # Wait for all futures to complete and gather results
    # results = [future.result() for future in futures]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        ocr_flow.serve(
            name="kiebids-ocr-deployment",
            parameters={},
        )
        # prefect deploy
    else:
        ocr_flow()
