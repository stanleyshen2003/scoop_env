#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf "$SCRIPT_DIR/question"
rm "$SCRIPT_DIR/answer.txt"
rm "$SCRIPT_DIR/lap_affordance.txt"
touch "$SCRIPT_DIR/answer.txt"
touch "$SCRIPT_DIR/lap_affordance.txt"
mkdir -p "$SCRIPT_DIR/question/image"
mkdir -p "$SCRIPT_DIR/question/text/system"
mkdir -p "$SCRIPT_DIR/question/text/user"