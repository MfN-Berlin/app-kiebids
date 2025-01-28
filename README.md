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
3. Debug

## Quickstart with example image
### Prerequisites
<!-- TODO GPU Support Cuda und  -->

Tested on Ubuntu 22.04 distribution
<!-- ffmpeg installation -->
Install ffmpeg library:
```
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 curl libcurl4
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
source .example.env
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
This starts the Prefect service, and you can view the dashboard at the url shown in the terminal (http://127.0.0.1:{port}).
To run the flow, you need to set the **image_path** in [workflow_config](./configs/workflow_config.yaml) to point to a folder with images.
Pipeline will loop through all images and will output xml files to the ```output_path``` defined in the config.

Behaviour:
- This will start a flow run on all images inside the `image_path` referenced in [workflow_config.yaml](./configs/workflow_config.yaml)

## Evaluation Modus (Work in Progress)
To enable evaluation, you need to set the following in [workflow_config](./configs/workflow_config.yaml):
```
evaluation: true
xml_path: "path/to/ground/truth/xml_files"
```
and optionally set ```run_id``` if you want to tag the evaluation with a specific name. This starts a tensorboard session where results from the modules is stored:

- Layout analysis: average iou
- Text recognition: average CER

The results for each run will be stored in the folder ```data/evaluation/tensorboard/{name_of_run}```. To view the dashboard, run:
```bash
tensorboard --logdir data/evaluation/tensorboard/{name_of_run}
```
The tensorboard updates every 1 minute during the pipeline process.

## Debug Modus
To enable debug mode, set ```mode: debug``` in the [workflow_config](./configs/workflow_config.yaml) file, and optionally ```run_id``` if you want to tag the debug run with a specific name.

Debug modus saves interim results after each module. You find the debug results from each module in the ```data/debug/{module}/{name_of_run}``` path.

The debug mode also serves a FiftyOne app at the end of the flow at the shown URL, where you can view the images.

### Run flow on own images:
You can either put more images inside the `data/images` directory or you can reference a directory on your system under => `image_path` in [workflow_config.yaml](./configs/workflow_config.yaml) (Make sure that you also adjust the `max_images` field to analyse the desired number of images)

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


## Known issues

### Prefect

**Database locked**

Prefect uses a SQLite database under the hood, and to ensure correct concurrent access the database occasionally locks. This causes the following error:

```bash
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked
```

This is is is not a dangerous error, and any requests to the database will simply retry until connection is established. This is a know issue from prefect: https://github.com/PrefectHQ/prefect/issues/10188

**Port already in Use / Connection refused**

After finishing running the pipeline, prefect will block the used port for a couple of minutes. If you start a run again within that time, you will get a connection error since the port is already in use:

```bash
Port xxxx is already in use. Please specify a different port with the `--port` flag.
```

Or the error:

```bash
httpx.ConnectError: [Errno 111] Connection refused
```

You can either wait a few minutes and try again, or set a new port number in the ```.env``` file.

### FiftyOne Database

When running the pipeline in debug mode the fiftyone database is enabled. If the pipeline was abruptly cancelled it may cause issues with the storing of data to the database. If such issues are encountered, you can simply delete the database before running the pipeline again:

```bash
rm data/debug/fifty-db --r
```