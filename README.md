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

## Prerequisites
<!-- ffmpeg installation -->
<!-- Files and models -->
<!-- ## Usage
1. Adapt [workflow_config.yaml](./configs/workflow_config.yaml) to your needs.
   e.g., set `image_path` to the path of your input images, etc.
2. Make a folder called `models` in the root directory (next to `data` etc.) and put the [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) model there.
3. Follow the [installation instructions](#installation) for your preferred method.
4. Run the workflow.
5. Inspect the results – PAGE XML files by default, images when in debug mode. -->

### Local Python environment
<!-- TODO try to run without conda -->
Install conda and create an environment:

See [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for further information on installing conda.
```bash
conda env create --file environment.yml
conda activate app-kiebids
```

**Once you set up a virtual environment and installed the dependencies, you can run the application by executing the following command in your terminal:**

```bash
bash run_flow.sh
```
This starts the Prefect service, and you can view the dashboard at: http://127.0.0.1:{port}
To run the flow, you need to set the following in [workflow_config](./configs/workflow_config.yaml):
- **image_path**: Path to folder with images

Pipeline will loop through all images found in ```image_path``` and output xml files to the ```output_path``` defined in the config.

## Evaluation Modus (Work in Progress)
To enable evaluation, you need to set the following in [workflow_config](./configs/workflow_config.yaml):
```
evaluation: true
xml_path: "path/to/ground/truth/xml_files"
```
and optionally set ```run_id``` if you want to tag the evaluation with a specific name. This starts a tensorboard session where results from the modules is stored:

- Layout analysis: average iou
- Text recognition: average CER

To view evaluation tensorboard, run: (you can see all previous runs under the below folder path)
```bash
tensorboard --logdir data/evaluation/tensorboard/{name_of_run}
```
The tensorboard updates every 1 minute during the pipeline process.

## Debug Modus
To enable debug mode, set ```mode: debug``` in the [workflow_config](./configs/workflow_config.yaml) file, and optionally ```run_id``` if you want to tag the debug run with a specific name.

Debug modus saves interim results after each module. You find the debug results from each module in the ```data/debug/{module}/{name_of_run}``` path.

The debug mode also serves a FiftyOne app at the end of the flow at the shown URL, where you can view the images.


### Dockerized application
Make sure you have `docker` and `docker compose` installed and Docker is running on your machine.
See [docker installation guide](https://docs.docker.com/get-docker/) for further information.

Please checkout the [dockerization branch](https://github.com/MfN-Berlin/app-kiebids/tree/dockerization?tab=readme-ov-file#run-with-docker) to launch the application via docker. `git checkout dockerization`
> The state of `dockerization branch` might be behind the `main` branch due to ongoing development process.

## Evaluation
To view evaluation tensorboard, run:
```bash
tensorboard --logdir data/evaluation/tensorboard/{name_of_run}
```
The tensorboard updates every 1 minute during the pipeline process.

## Testing (WIP)

Run pytests:
```bash
pytest -s
```

## Development Environment KI-IW
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

## FiftyOne Database

When running the pipeline in debug mode (mode: debug in dev_workflow.yaml) the fiftyone database is enabled. If the pipeline was abruptly cancelled it may cause issues with the storing of data to the database. If such issues are encountered, you can simply delete the database before running the pipeline again:

```bash
rm data/debug/fifty-db --r
```