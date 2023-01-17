#!/bin/bash

conda activate adelai
# sudo $(which python3) LeReS/tools/test_shape.py
# nohup sudo $(which python3) LeReS/tools/test_shape.py &
# tail -f nohup.out
nohup sudo $(which python3) app.py >out.log 2>out.err < /dev/null &
