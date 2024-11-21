# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

## Python environment setup

Create conda environment and install dependencies:
```
conda create -n zug-mfn python=3.10.13
conda activate zug-mfn
pip install -r requirements.txt
```

Install pre-commit hooks:
```
pip install pre-commit
pre-commit install
```

## Usage

### Run with docker
> **Tested on macos with M1 Chip and Docker Desktop v4.35.1**

1. [Download necessary models](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) into [./models](./models/) directory. 
<!-- only sam model downloaden? -->
<!-- TODO put one image as test image -->
2. One test image is available inside `data/images` directory. You are free to add more images into this directory. As for now only 10 images will be processed. If you would like to process more just change the `max_images` parameter inside [docker_config.yml](./configs/docker_config.yml)
3. Run the containers to serve prefect and workflow (prefect deployments):
    ```
    docker-compose -f docker/docker-compose.yml up --build
    ```
    At first execution this will download further necessary models.
4. Wait until the `kiebids_ocr` container started (indicated by message `You can also run your flow via the Prefect UI ...`) and open the prefect UI in your browser `http://0.0.0.0:4200/`
5. On the left sidebar click on `Deployments` and select the `KIEBIDS deployment`
6. Click upper right button `Run` and select `Quick run`

Behaviour:
- This will trigger a deployment run
- You can follow the progress in your terminal for more detailed logs.
- The produced results of each respective module can be inspected inside the `data/debug` directory. 

### Run ocr flow locally

Create .env file containing port env variable:
```
PREFECT_PORT=<some-port-number-between-4200-and-4300>
```
Run:
```
bash run_flow.sh
```
This will start the prefect server in background (if not started so far) and execute the basic flow.

If you'd like to kill the prefect server at execution end then run:
```
bash run_flow.sh --stop_prefect
```

To start the server with a deployment (https://docs.prefect.io/3.0/deploy/index) you can run:
```
bash run_flow.sh --serve-deployment
```

### Observe Debugging Results in FiftyOne App

Set ocr flow to debug mode inside [the config file](./configs/default_config.yml). After processing a fiftyone app is served at displayed url. It persists previous results of each module for each given image.
You can also run only the app to inspect your previous runs by running
```
python kiebids/ocr_flow.py --fiftyone-only
```

You can inspect the results for each image by filtering the `image_name` field inside the app.

> This tracking is currently activated only in debug mode
----
### Run pytest:

```
pytest -s
```

-----



## Modules
Overview of each module

1. Preprocessing
2. Layout Analysis
3. Text Recognition
4. Semantic Labeling
5. Entity linking
6. Evaluation
