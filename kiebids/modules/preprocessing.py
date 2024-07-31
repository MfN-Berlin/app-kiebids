import os
import shutil

from prefect import task


@task
def preprocessing(image_path, output_path):
    # dummy task
    # copy image to output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(image_path, output_path)
