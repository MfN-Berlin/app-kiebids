#!/bin/bash

usage() {
    echo "Usage: $0 [--serve-deployment] [--continue-prefect] [--help]"
    echo
    echo "Options:"
    echo "  --serve-deployment    serve the prefect deployment"
    echo "  --continue-prefect     continue the prefect server after running the flow"
    echo "  --help        Display this help message."
    exit 1
}
serve_deployment=0
continue_prefect=0

# Parse and process command-line options manually for cross-platform compatibility
while [[ $# -gt 0 ]]; do
    case "$1" in
        --serve-deployment)
            serve_deployment=1
            shift
            ;;
        --continue-prefect)
            continue_prefect=1
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
            echo "Error: Invalid option: $1"
            usage
            ;;
    esac
done

export PYTHONPATH=$PYTHONPATH:.

if [ -z "$PREFECT_PORT" ]; then
    echo "defaulting PREFECT_PORT to 4200"
    PREFECT_PORT=4200
fi

export PREFECT_API_URL=http://localhost:$PREFECT_PORT/api

# Function to gracefully stop the first process
stop_prefect() {
    PREFECT_PID=$(lsof -i :$PREFECT_PORT | awk 'NR>1 {print $2}')

    if [ -n "$PREFECT_PID" ]; then
        echo "Stopping prefect process..."
        kill $PREFECT_PID
    fi
    exit 0
}

# Trap the keyboard interrupt (Ctrl+C) to stop processes gracefully
trap 'stop_prefect' INT

# Get the path to the current Python executable
INTERPRETER_PATH=$(which /usr/bin/env python)

# Check if port $PREFECT_PORT is in use
if ! lsof -i :$PREFECT_PORT > /dev/null; then
    prefect server start --host localhost --port $PREFECT_PORT &

    # wait for prefect server to start
    sleep 5
fi

# Finally, run the flow
if [ $serve_deployment -eq 1 ]; then
    $INTERPRETER_PATH kiebids/ocr_flow.py --serve-deployment
else
    $INTERPRETER_PATH kiebids/ocr_flow.py
    if [ $continue_prefect -eq 0 ]; then
        stop_prefect
    fi
fi
