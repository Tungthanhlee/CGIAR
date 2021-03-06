#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
*__pycache__*
*.ipynb_checkpoints*
outputs*
runs*
weights*
.git/*
cache/*
logs/*
*.md
*.txt
*.ipynb
push_code.sh
*submission

EOM

USER="dgx2"
# push code to server
rsync -vr --exclude-from $TEMP_FILE . $USER:/data/tung/CIGAR/src/
# pull model weights and log files from server
# rsync -vr --exclude-from $TEMP_FILE . $USER:/data/datasets/LNDB_segmentation/runs
# rsync -vr $USER:$REMOTE_HOME/tung_chexpert/Chest-Radiograph-Interpretation-DL/experiments/nasnet_cardiomegaly/ ./experiments/nasnet_cardiomegaly/
rm $TEMP_FILE
