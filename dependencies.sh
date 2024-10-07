#!/bin/bash

eval "$(command conda "shell.bash" "hook" 2> /dev/null)"
eval "$(command conda "shell.zsh" "hook" 2> /dev/null)"

conda env create -f environment.yml
conda activate cyclogeostrophy_impact_experiment
pip install --upgrade pip
conda install pyinterp
pip install -r requirements.txt --upgrade