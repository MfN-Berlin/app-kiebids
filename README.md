# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

## Python environment setup
```
conda create -n zug-mfn python=3.10.13
conda activate zug-mfn
pip install -r requirements.txt
```

## Usage

### Run ocr flow locally
Run: 
```
bash run_flow.sh
```
This will start the prefect server in background (if not started so far) and execute the basic flow.

If you'd like to kill the prefect server at execution end then run:
```
bash run_flow.sh --stop_prefect
```

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
