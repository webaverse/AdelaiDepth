#!/bin/bash

conda activate adelai
# sudo $(which python3) LeReS/tools/test_shape.py
nohup sudo $(which python3) LeReS/tools/test_shape.py &
tail -f /home/ubuntu/nohup.out