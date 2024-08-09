import os 

import pytesseract
from pathlib import Path
from prefect import task


@task
def text_recognition(image_name, input_path, output_path, debug=False):

    OUTPUT_DIR_TEXT_RECOGNITION = Path(output_path) / "text_recogniton"
    
    # Get all cropped images related to the input image
    images = [image for image in os.listdir(input_path) if image.startswith(image_name)]

    for image in images: 
        image_path = Path(input_path) / image
        text = get_text(image_path)

        text_output_path =  OUTPUT_DIR_TEXT_RECOGNITION /  os.path.basename(image_path).split(".")[0] + '.txt'

        with open(text_output_path, 'w') as f:
                f.write(text)


def get_text(image_path, debug=False):
    '''
    Get ocr text from image with tesseract
    '''
    text = pytesseract.image_to_string(image_path)
    return text 