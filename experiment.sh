#!/bin/bash

# Default values for optional variables
LOG_ROOT="experiment_log"  # Default path to experiment logs
MAX_TRIALS=5  # Default number of trials
CHECK_INTERVAL=5  # Time interval between checks (in seconds)

# Function to display usage
usage() {
    echo "Usage: $0 -e <EXP_ID> -c <CONFIG_FILE> [-l <LOG_ROOT>] [-t <MAX_TRIALS>]"
    echo "  -e EXP_ID: Required. Experiment ID."
    echo "  -c CONFIG_FILE: Required. Path to your config file."
    echo "  -l LOG_ROOT: Optional. Path to experiment logs (default: $LOG_ROOT)."
    echo "  -t MAX_TRIALS: Optional. Number of trials to perform (default: $MAX_TRIALS)."
    echo "  -h: Show this help message."
    exit 1
}

# Parse flags
while getopts "e:c:l:t:h" opt; do
    case $opt in
        e) EXP_ID=$OPTARG ;;
        c) CONFIG_FILE=$OPTARG ;;
        l) LOG_ROOT=$OPTARG ;;  # Override default LOG_ROOT if provided
        t) MAX_TRIALS=$OPTARG ;;  # Override default MAX_TRIALS if provided
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure required flags are provided
if [ -z "$EXP_ID" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Error: Both -e (EXP_ID) and -c (CONFIG_FILE) are required."
    usage
fi

# Path to results folder
RESULT_DIR="$LOG_ROOT/$EXP_ID"

# Export environment variables for other scripts
export RESULT_DIR=$RESULT_DIR
export CONFIG_FILE=$CONFIG_FILE

# Prepare the experiment
echo "Preparing the experiment..."
python prepare_experiment.py

# Trial loop
for ((trial=1; trial<=MAX_TRIALS; trial++)); do
    echo "Trial $trial of $MAX_TRIALS: Checking results..."

    all_done=true
    for folder in "$RESULT_DIR"/*; do
        done=false
        for subfolder in "$folder"/*; do
            if [ -d "$subfolder" ]; then  # Check if it's a directory
                if [ -f "$subfolder/result_sequence.txt" ]; then
                    done=true
                    break 
                fi
            fi
        done
        if ! $done; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo "All experiments are complete. Exiting..."
        ./zip_result.sh
        exit 0
    fi
    
    python main.py

    if [ $trial -lt $MAX_TRIALS ]; then
        echo "Waiting for $CHECK_INTERVAL seconds before the next trial..."
        sleep $CHECK_INTERVAL
    fi
done

echo "Reached maximum number of trials ($MAX_TRIALS). Exiting..."
exit 1
