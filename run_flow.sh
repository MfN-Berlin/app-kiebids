#!/bin/bash

usage() {
    echo "Usage: $0 [--serve-deployment] [--stop-prefect] [--help]"
    echo
    echo "Options:"
    echo "  --serve-deployment    serve the prefect deployment"
    echo "  --stop-prefect     stop the prefect server after running the flow"
    echo "  --help        Display this help message."
    exit 1
}
serve_deployment=0
stop_prefect=0

# Parse command-line options using getopt
OPTIONS=$(getopt -o "" --long "serve-deployment,stop-prefect,help" -- "$@")
if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$OPTIONS"

# Process the arguments
while true; do
    case "$1" in
        --serve-deployment)
            serve_deployment=1
            shift
            ;;
        --stop-prefect)
            stop_prefect=1
            shift
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: Invalid option"
            echo $1
            usage
            ;;
    esac
done

export PYTHONPATH=$PYTHONPATH:.
source .env

if [ -z "$PREFECT_PORT" ]; then
    echo "PREFECT_PORT is not set. please define PREFECT_PORT in .env file"
    exit 1
fi

export PREFECT_API_URL=http://localhost:$PREFECT_PORT/api

# Function to gracefully stop the first process
stop_prefect() {
    echo "Stopping prefect process..."
    # kill $PREFECT_PROCESS_PID  # Kill the first process using its PID
    kill $(lsof -i :$PREFECT_PORT | awk 'NR>1 {print $2}')
    exit 0
}

# Trap the keyboard interrupt (Ctrl+C) to stop processes gracefully
trap 'stop_prefect' INT

# Get the path to the current Python executable
PYTHON_PATH=$(which /usr/bin/env python)

# Check if port $PREFECT_PORT is in use
if ! lsof -i :$PREFECT_PORT > /dev/null; then
    prefect server start --port $PREFECT_PORT &
    # Capturing PID of prefect server
    PREFECT_PROCESS_PID=$!

    # wait for prefect server to start
    sleep 5
fi

# Finally, run the flow
if [ $serve_deployment -eq 1 ]; then
    $PYTHON_PATH kiebids/ocr_flow.py --serve-deployment
else
    $PYTHON_PATH kiebids/ocr_flow.py
    if [ $stop_prefect -eq 1 ]; then
        stop_prefect
    fi
fi
