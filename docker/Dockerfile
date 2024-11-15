FROM prefecthq/prefect:2-python3.10
WORKDIR /app

# Install Tesseract OCR dependencies and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6
#     tesseract-ocr \
#     libtesseract-dev \
#     libleptonica-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./data /app/data
COPY ./kiebids /app/kiebids
COPY ./configs /app/configs
COPY ./configs/docker_config.yml /app/configs/default_config.yml

# TODO put this to the other installs
RUN apt-get install -y curl libcurl4

CMD ["python", "kiebids/ocr_flow.py", "--serve-deployment"]