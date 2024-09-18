import os
import sys
from pathlib import Path

from kiebids.modules.layout_analysis import LayoutAnalyzer
from kiebids.modules.preprocessing import preprocessing
from kiebids.modules.text_recognition import text_recognition

# commented out for now to avoid tensorflow loading
# from modules.semantic_labeling import semantic_labeling
from prefect import flow

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = Path(os.environ.get("INPUT_DIR", BASE_DIR / "data" / "input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "data" / "output"))


@flow(name="KIEBIDS pipeline", log_prints=True)
def ocr_flow():

    image_paths = [
        os.path.join(INPUT_DIR, file)
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif"))
    ]

    # init objects/models for every stage
    layout_analyzer = LayoutAnalyzer()

    # Process images sequentially
    for image_path in image_paths:
        # accepts image path. outputs image
        preprocessed_image = preprocessing(image_path=image_path)

        # accepts image. outputs image and bounding boxes. if debug the write snippets to disk
        bb_labels = layout_analyzer.run(preprocessed_image)

        # text_recognition.run
        # accepts image and bounding boxes. returns. if debug the write snippets with corresponding text to disk
        text_recognition_output_dir = text_recognition(preprocessed_image, bb_labels)

        # semantic_labeling.run
        # semantic_labeling_output_dir = semantic_labeling(layout_analysis_output_dir, output_path)
        # entity_linking.run
        # entity_linking(image_path, output_path)

    # # Process images concurrently
    # futures = process_single_image.map(image_paths, OUTPUT_DIR)

    # # Wait for all futures to complete and gather results
    # results = [future.result() for future in futures]


if __name__ == "__main__":
    ocr_flow()
