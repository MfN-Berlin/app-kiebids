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


## Usage

1. Adapt [workflow_config.yaml](./configs/workflow_config.yaml) to your needs.
   e.g., set `image_path` to the path of your input images, etc.
2. Make a folder called `models` in the root directory (next to `data` etc.) and put the [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) model there.
3. Follow the [installation instructions](#installation) for your preferred method.
4. Run the workflow.
5. Inspect the results – PAGE XML files by default, images when in debug mode.

## Installation

There are two ways to run this application:

1. From the command line in a local Python environment that will run the workflow automatically.
2. As a dockerized application which allows you to start workflow runs from the Prefect frontend.

The dockerized variant is preferred in the long run but is not yet fully functional.
For the time being, use the local variant.

### Local Python environment

Set up a virtual environment using your preferred Python management tool.

**barebones `venv` example:** Make sure you have Python 3.10(.13) installed.
```bash
python3.10 -m venv app-kiebids
source app-kiebids/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**`conda` example:**
See [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for further information on installing conda.
```bash
conda create -n app-kiebids python=3.10.13
conda activate app-kiebids
pip install -U pip
pip install -r requirements.txt
conda install -c conda-forge pyvips
```

**`uv` example:**
(See [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for further information on installing uv.)

```bash
uv venv --python 3.10.13
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt
```

**Once you set up a virtual environment and installed the dependencies, you can run the application by executing the following command in your terminal:**

```bash
bash run_flow.sh
```

This will start the prefect server in background (if not started so far) and execute the basic flow.
For further script options, see:
```
bash run_flow.sh --help
```


**Alternatively you can start the work flow by running the `ocr_flow.py` script directly:**

```bash
prefect cloud login  # if not already logged in; will provide a link to log in
python kiebids/ocr_flow.py
```


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

## Testing

Run pytests:
```bash
pytest -s
```

## Development Mode

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
