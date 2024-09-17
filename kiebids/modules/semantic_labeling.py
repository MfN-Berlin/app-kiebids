import os
import shutil

import tensorflow as tf
import numpy as np

from prefect import task
from pathlib import Path


script_path = Path(__file__).parent.resolve()


@task
def semantic_labeling(image_dir, output_path, debug=False):
    OUTPUT_DIR_SEMANTIC_LABELING = Path(output_path) / "semantic_labeling"
    os.makedirs(OUTPUT_DIR_SEMANTIC_LABELING, exist_ok=True)

    images = [image for image in os.listdir(image_dir)]

    # TODO: this is the handwritten/not handwritten classifier, loop for all the models
    model_path = (
        script_path.parent.parent
        / "models"
        / "semantic_labeling"
        / "label_classifier_hp"
    )

    print("Loading model: ", model_path)
    model = load_model(model_path)
    class_names = ["handwritten", "printed"]

    for class_name in class_names:
        class_dir = OUTPUT_DIR_SEMANTIC_LABELING / class_name
        os.makedirs(class_dir, exist_ok=True)

    for image in images:
        image_path = Path(image_dir) / image

        # TODO: Do something with the score from the prediction
        entry = predict_label(image_path, model, class_names=class_names)
        shutil.copy(
            image_path, str(OUTPUT_DIR_SEMANTIC_LABELING / entry["class"] / image)
        )


def load_model(model_path):
    """
    Load a trained Keras Sequential image classifier model.

    Args:
        path_to_model (str): Path to the model file.

    Returns:
        model (tf.keras.Sequential): Trained Keras Sequential image classifier model.
    """
    model = tf.keras.models.load_model(model_path)
    return model


def predict_label(image_path, model, class_names=["handwritten", "printed"]):
    """
    Predict the label of an image using a trained image classifier model.
    """
    img_width = 180
    img_height = 180

    image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    entry = {}
    entry["filename"] = os.path.basename(
        image_path
    )  # Get the filename without the directory
    entry["class"] = class_names[np.argmax(score)]
    entry["score"] = 100 * np.max(score)

    return entry
