# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

[Insert description]

## Modules and modes

Overview of each module

1. Preprocessing
2. Layout Analysis
3. Text Recognition
4. Semantic Labeling
5. Entity linking

The pipeline can be run in three different modes:
1. Prediction
2. Evaluation
3. Debugging


## Usage

1. Adapt [workflow_config.yaml](./configs/workflow_config.yaml) to your needs.
   e.g., set `image_path` to the path of your input images, etc.
2. Follow the installation instructions for your preferred method.
3. Run the workflow.
4. Inspect the results – PAGE XML files by default, images when in debug mode.


## Installation

There are two ways to run this application:

1. as dockerized application which allows you to start workflow runs from the prefect frontend.
2. from the command line in a local Python environment that will run the workflow automatically.


### Dockerized application (preferred)

Make sure you have `docker` and `docker compose` installed and Docker is running on your machine.
See [docker installation guide](https://docs.docker.com/get-docker/) for further information.

TODO: Add instructions for installing `docker compose`.

**Start the application by running the following command in your terminal:**

```bash
docker compose up
```

**You can now access the frontend at [http://localhost:4200](http://localhost:4200).**

**To stop the application:**
```bash
docker compose stop  # stops the application
docker compose down  # stops the application and removes the containers
```

### Local Python environment (developers)

Set up a virtual environment using your preferred Python management tool.

**`conda` example:**
See [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for further information on installing conda.
```bash
conda create -n app-kiebids python=3.10.13
conda activate app-kiebids
pip install -U pip
pip install -r requirements.txt
```

**`uv` example:**
(See [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for further information on installing uv.)

```bash
uv venv --python 3.10.13
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt
```

**Now you can run the application by executing the following command in your terminal:**

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


## Testing

Run pytests:
```bash
pytest -s
```

## Debugging

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
