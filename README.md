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

<!-- TODO -->
> Behaviour / what to expect:


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
