from modules.preprocessing import preprocessing
from modules.layout_analysis import layout_analysis
from modules.text_recognition import text_recognition
from modules.semantic_labeling import semantic_labeling
from modules.entity_linking import entity_linking


def process_single_image():

    preprocessing(image)
    layout_analysis(image)
    text_recognition(image)
    semantic_labeling(image)
    entity_linking(image)


def ocr_flow():
    
    INPUT_DIR = "../data/"
  
    image_files = [
        os.path.join(INPUT_DIR, file)
        for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif"))
    ]

    for image_file in image_files:
        process_single_image(image_file)


if __name__ == "__main__":
    ocr_flow()
