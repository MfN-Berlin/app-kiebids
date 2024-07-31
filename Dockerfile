FROM prefecthq/prefect:2-python3.10

WORKDIR /app

# Install Tesseract OCR dependencies and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    libglib2.0-0

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV INPUT_DIR=/app/data/input
ENV OUTPUT_DIR=/app/data/output