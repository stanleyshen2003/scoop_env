#!/bin/bash

# Default values for optional variables
LOG_ROOT="experiment_log"  # Default path to experiment logs
MAX_TRIALS=5  # Default number of trials
CHECK_INTERVAL=5  # Time interval between checks (in seconds)
NOW_TIME=$(LC_TIME=en_US.utf8 date +%Y%m%d_%H%M%S)
# Function to display usage
usage() {
    echo "Usage: $0 -e <EXP_ID> -c <CONFIG_FILE> -n <TEST_TYPE> [-l <LOG_ROOT>] [-t <MAX_TRIALS>]"
    echo "  -e EXP_ID: Required. Experiment ID."
    echo "  -c CONFIG_FILE: Required. Path to your config file."
    echo "  -n TEST_TYPE: Required. Type of test to run."
    echo "  -l LOG_ROOT: Optional. Path to experiment logs (default: $LOG_ROOT)."
    echo "  -t MAX_TRIALS: Optional. Number of trials to perform (default: $MAX_TRIALS)."
    echo "  -h: Show this help message."
    exit 1
}

# Parse flags
while getopts "e:c:n:l:t:h" opt; do
    case $opt in
        e) EXP_ID=$OPTARG ;;
        c) CONFIG_FILE=$OPTARG ;;
        n) TEST_TYPE=$OPTARG ;;
        l) LOG_ROOT=$OPTARG ;;  # Override default LOG_ROOT if provided
        t) MAX_TRIALS=$OPTARG ;;  # Override default MAX_TRIALS if provided
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure required flags are provided
if [ -z "$EXP_ID" ] || [ -z "$CONFIG_FILE" ] || [ -z "$TEST_TYPE" ]; then
    echo "Error: Both -e (EXP_ID), -c (CONFIG_FILE), and -c (TEST_TYPE) are required."
    usage
fi

# Path to results folder
RESULT_DIR="${LOG_ROOT}/${TEST_TYPE}_${NOW_TIME}_${EXP_ID}"
echo "Results will be saved in $RESULT_DIR"

# Export environment variables for other scripts
export RESULT_DIR=$RESULT_DIR
export CONFIG_FILE=${CONFIG_FILE:-src/config/config.yaml}
export TEST_TYPE=$TEST_TYPE


CONFIGURATIONS=$(yq 'to_entries | .[:] | map(.key as $parent | .value | to_entries | .[:] | map([$parent, .key])) | flatten' $CONFIG_FILE | sed '/^#/d; s/ #.*//' | sed 's/- //')
config_array=($CONFIGURATIONS)

# Prepare the experiment
echo "Preparing the experiment..."
python prepare_experiment.py

for ((i = 0; i < ${#config_array[@]}; i+=2)); do
    export TASK_TYPE=${config_array[i]}
    export ENV_IDX=${config_array[i+1]}
    echo "Running task $TASK_TYPE with environment $ENV_IDX"
    python main.py
done

# echo "Reached maximum number of trials ($MAX_TRIALS). Exiting..."

echo "Convert codec of video"
python convert_codec.py $RESULT_DIR
exit 1
