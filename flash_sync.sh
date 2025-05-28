#!/bin/bash

# dataset structure
# ├── 4dhoi
# │   ├── xx_xx_xx(date)
# │   │   ├── personxx
# │   │   │     ├── avatar
# │   │   │     ├── motionxx
# │   ├── xx_xx_xx(date)
# │   │   ├── personxx
# │   │   │     ├── avatar
# │   │   │     ├── motionxx

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/apps/easybuild/software/Anaconda3/2024.02-1/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/easybuild/software/Anaconda3/2024.02-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate portrait_magic 

ROOT_DIR=/nas_data/dataset/4dhoi/25_04_03/person05
PYTHON_SCRIPT=/nas_data/home/hlp/code/4dhoi/HOI_Gen/pre_processing/flash.py

find "$ROOT_DIR" -type d -name "avatar" | while read -r avatar_dir; do
    echo "Processing avatar directory: $avatar_dir"
    python "$PYTHON_SCRIPT" --folder "$avatar_dir" 
done

find "$ROOT_DIR" -type d -name "motion*" | while read -r motion_dir; do
    echo "Processing motion directory: $motion_dir"
    python "$PYTHON_SCRIPT" --folder "$motion_dir"
done