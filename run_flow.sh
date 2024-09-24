#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# Function to gracefully stop the first process
stop_prefect() {
    echo "Stopping prefect process..."
    # kill $PREFECT_PROCESS_PID  # Kill the first process using its PID
    kill $(lsof -i :4200 | awk 'NR>1 {print $2}')
    exit 0
}

# Trap the keyboard interrupt (Ctrl+C) to stop processes gracefully
trap 'stop_prefect' INT

# Get the path to the current Python executable
PYTHON_PATH=$(which /usr/bin/env python)

# Check if port 4200 is in use
if ! lsof -i :4200 > /dev/null; then
    # prefect server start & sleep 5 && $PYTHON_PATH kiebids/ocr_flow.py
    prefect server start &
    # Capturing PID of prefect server
    PREFECT_PROCESS_PID=$!

    # wait for prefect server to start
    sleep 5
fi

# wait for flow to finish
$PYTHON_PATH kiebids/ocr_flow.py

if  [ "$1" = "--stop_prefect" ]; then
    stop_prefect
fi
