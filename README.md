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
