# Anwendungsworkflow zur Informationsextraktion aus Sammlungsetiketten

## Usage

```
# 1/ Run kiebids/ocr_flow.py locally
prefect cloud login
python kiebids/ocr_flow.py

# 2/ Build images and start services defined in compose.yaml.
#    Access web server at http://127.0.0.1:4200. You can trigger a pipeline run
#    by going to deployments, clicking on the three dots, choosing Quick Run.
docker compose up --build
# stop and remove running services
docker compose down
```


## Teuxdeux

- Reuse Docker image