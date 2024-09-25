#!/bin/bash

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

# TODO adjust this when running with deployment server
# wait for flow to finish
$PYTHON_PATH kiebids/ocr_flow.py

if  [ "$1" = "--stop_prefect" ]; then
    stop_prefect
fi
