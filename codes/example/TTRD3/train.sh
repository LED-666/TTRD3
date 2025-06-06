#!/bin/sh

ROOT=../../..
export PYTHONPATH=$ROOT:$PYTHONPATH

python -u main.py options/TTRD3.yml --mode train