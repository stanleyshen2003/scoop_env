#!/bin/bash

SUBROOT_DIR=$RESULT_DIR
ZIP_FILE="$(basename "$SUBROOT_DIR").zip"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEST_DIR="$SCRIPT_DIR/result"

if [ -z "$SUBROOT_DIR" ]; then
    echo "Error: Please provide the subroot directory name as an argument."
    exit 1
fi

if [ ! -d "$SUBROOT_DIR" ]; then
    echo "Error: Directory $SUBROOT_DIR does not exist."
    exit 1
fi

cd "$(dirname "$SUBROOT_DIR")" || exit 1
zip -r "$ZIP_FILE" "$(basename "$SUBROOT_DIR")"

mv "$ZIP_FILE" "$DEST_DIR"