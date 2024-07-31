# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

## Usage

```
# 1/ Run kiebids/ocr_flow.py locally
prefect cloud login
python kiebids/ocr_flow.py

# 2/ Run kiebids/ocr_flow.py in a local Docker container
#    This will execute the flow /pipeline, then quit.
docker compose --profile local up --build -d

# 3/ Run kiebids/ocr_flow.py, then trigger flows over at http://127.0.0.1:4200
docker compose --profile server up --build -d

# 4/ Build an image, start a container and drop into a bash, do whatever you want
docker compose build kiebids_ocr_interactive
docker run -it --rm --entrypoint /bin/bash kiebids-ocr
```


## Teuxdeux

- Reuse Docker image