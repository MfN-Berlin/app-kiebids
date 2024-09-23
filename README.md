# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

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

-----

### 2/ Build images and start services defined in compose.yaml.
#    Access web server at http://127.0.0.1:4200. You can trigger a pipeline run
#    by going to deployments, clicking on the three dots, choosing Quick Run.
```
docker compose up --build
# stop and remove running services
docker compose down
```

----- 
## Teuxdeux

- Reuse Docker image



## Modules
Overview of each module 

1. Preprocessing 
2. Layout Analysis 
3. Text Recognition
4. Semantic Labeling
5. Entity linking 
6. Evaluation


## Data Structure 

The output for each module will fall into the following folder structure: 

```
── data

│   ├── output
│   │   ├── preprocessing
│   │   │   ├── image_X.jpg
│   │   ├── layout_analysis
│   │   |   ├── image_X
│   │   |   │   ├── image_X_1.jpg
│   │   |   │   ├── image_X_2.jpg
│   │   ├── text_recognition 
│   │   |   ├── image_X
│   │   |   │   ├── image_X_1_ocr.txt
│   │   |   │   ├── image_X_2_ocr.txt
│   │   ├── semantic_labeling
│   │   |   │   ├── image_X_1_handwritten
│   │   |   │   ├── image_X_1_barcode 
│   │   ├── entity_linking
``` 

For each module there is an optional 'debug' flag. If the debug flag is set to true, interim results will be saved to the ```data/debug``` folder, where each module has its own subfolder. 
```
│   ├── debug 
│   │   ├── preprocessing
│   │   ├── layout_analysis
│   │   ├── text_recocgnition 
│   │   ├── semantic_labelling
│   │   ├── entity_linking
```

