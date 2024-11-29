FROM prefecthq/prefect:3-python3.10
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 curl libcurl4

COPY ./docker/docker_requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
Run pip install "prefect[docker]"

COPY ./kiebids /app/kiebids
COPY ./configs /app/configs

CMD ["python", "kiebids/ocr_flow.py", "--serve-deployment"]
