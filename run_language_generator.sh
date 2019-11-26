#!/usr/bin/env bash
cur_path=$(pwd)

context=${1}

${cur_path}/main.py --input_words="${context}"
