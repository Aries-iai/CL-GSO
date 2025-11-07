#!/bin/bash

# init virtual environment
module load xx
source activate xx # add your own path here

# command to run
CUDA_VISIBLE_DEVICES=xx python generation.py --input_file "data/advbench_new.csv" --data_format "csv"



