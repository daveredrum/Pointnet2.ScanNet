#!/bin/bash

#SBATCH -p debug
#SBATCH -q normal
#SBATCH --job-name=train                           # Job name
#SBATCH --mail-type=BEGIN,END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhenyu.chen@tum.de             # Where to send mail
#SBATCH --mem=100gb                                 # Job memory request
#SBATCH --cpus-per-gpu=8                           # Job CPUs request
#SBATCH --gpus=rtx_3090:1

# #SBATCH --time=48:00:00                            # Time limit hrs:min:sec
#SBATCH --output=/rhome/dchen/Pointnet2.ScanNet/logs/%j.log      # Standard output and error log

# Default output information
date;hostname;pwd

# scripts
python scripts/train.py --use_multiview --use_normal --tag ssg
# python scripts/train.py --use_multiview --use_normal --use_msg --tag msg
