#!/bin/bash

source ~/.venv/bin/activate

kaggle datasets download -d polomarco/chest-ct-segmentation -p ./dataset --unzip