#!/usr/bin/env bash
cur_path=$(pwd)

context=${1}
topic=${2}

${cur_path}/main.py --input_words="${context}" --topic="${topic}"
