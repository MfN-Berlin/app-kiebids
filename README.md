# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

[Insert description]

## Modules and modes

Overview of each module

1. Preprocessing
2. Layout Analysis
3. Text Recognition
4. Semantic Labeling (not yet implemented)
5. Entity linking (not yet implemented)

The pipeline can be run in three different modes:
1. Prediction (work in progress)
2. Evaluation (work in progress)
3. Debugging

## Quickstart with example image
### Prerequisites
<!-- TODO GPU Support Cuda und  -->

Tested on Ubuntu 22.04 distribution
<!-- ffmpeg installation -->
Install ffmpeg library:
```
sudo apt update
sudo apt install ffmpeg
```

<!-- Files and models -->
After cloning this repository, download the required SAM Model by running:
```
cd ./app-kiebids
wget -P ./models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Set up local Python environment
<!-- TODO try to run without conda -->
Install conda (see [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) and create a local python environment by using the bash shell:
```bash
conda env create --file environment.yml
conda activate app-kiebids
```

### Run app and trigger flow run from browser
Once the dependencies are installed run:
```bash
bash run_flow.sh --serve-deployment
```

This will serve a self hosted prefect environment:
1. Copy and paste the shown url of respective deployment. The URL should look similar to this `http://localhost:4200/deployments/deployment/<some-random-deployment-id>`
2. Click upper right button `Run` and select `Quick run`

Behaviour:
- This will start a flow run on all images inside the `image_path` referenced in [workflow_config.yaml](./configs/workflow_config.yaml)
- You can follow the progress in your terminal for more detailed logs.

### Running a flow without prefect UI:
```bash
bash run_flow.sh
```

Behaviour:
- This will start a flow run on all images inside the `image_path` referenced in [workflow_config.yaml](./configs/workflow_config.yaml)

### Run flow on own images:
You can either put more images inside the `data/images` directory or you can reference a directory on your system under => `image_path` in [workflow_config.yaml](./configs/workflow_config.yaml) (Make sure that you also adjust the `max_images` field to analyse the desired number of images)

## Evaluation
To view evaluation tensorboard, run:
```bash
tensorboard --logdir data/evaluation/tensorboard/{name_of_run}
```
The tensorboard updates every 1 minute during the pipeline process.

## Dockerized application
Make sure you have `docker` and `docker compose` installed.
See [docker installation guide](https://docs.docker.com/get-docker/) for further information.

Please checkout the [dockerization branch](https://github.com/MfN-Berlin/app-kiebids/tree/dockerization?tab=readme-ov-file#run-with-docker) to run the application via docker. `git checkout dockerization`
> The state of `dockerization branch` might be behind the `main` branch due to ongoing development process.

## Testing (WIP)

Run pytests:
```bash
pytest -s
```

## Development Environment KI-Ideenwerkstatt
### Config behaviour

Inside the your local `.env` file (see [.example.env](.example.env)) set the following two variables to ensure that the development configs are initialized with paths to our shared directories.
```
OCR_CONFIG="dev_ocr_config.yaml"
WORKFLOW_CONFIG="dev_workflow_config.yaml"
```
If these variables are not set, the default [workflow_config](./configs/workflow_config.yaml) and [ocr_config](./configs/ocr_config.yaml) are initialized instead.

### Observe debugging results in the FiftyOne app

Set ocr flow to debug mode inside the [workflow config file](./configs/workflow_config.yaml).
After processing a fiftyone app is served at the displayed URL. It persists previous results of each module for each given image.
You can also run the app standalone to inspect your previous runs by running
```
python kiebids/ocr_flow.py --fiftyone-only
```

You can inspect the results for each image by filtering the `image_name` field inside the app.

> This tracking is currently activated only in debug mode

-----
