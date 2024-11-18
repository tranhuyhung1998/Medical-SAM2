#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=30           # Number of CPU to request for the job
#SBATCH --mem=100GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=01-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=project                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=cs701qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=hh.tran.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=hungth_sam2     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Anaconda3/2023.09-0
module load CUDA/12.4.0

# Create a virtual environment
# python3 -m venv ~/myenv
conda create -n yolo python=3.10

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
# source ~/myenv/bin/activate
conda activate yolo

# If you require any packages, install it as usual before the srun job submission.
# pip3 install numpy
# pip3 install -r requirements.txt
pip install ultralytics

# Submit your job to the cluster
# srun --gres=gpu:1 python /path/to/your/python/script.py

# srun --gres=gpu:1 python3 train_3d.py -net sam2 -exp_name Leaderboard_pred -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -pretrain ./checkpoints/MedSAM2_pretrain.pth -sam_config sam2_hiera_t -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 1 -dataset leaderboard -data_path ./data/Public_leaderboard_data -video_length 3
# srun --gres=gpu:1 python3 inference_3d.py -net sam2 -exp_name Leaderboard_pred -pretrain checkpoints/MedSAM2_pretrain -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -image_size 1024 -out_size 512 -val_freq 1 -prompt bbox -prompt_freq 1 -dataset leaderboard -data_path ./data/Public_leaderboard_data
# srun rm -r output/test_labels
# srun --gres=gpu:1 python3 inference_3d.py -net sam2 -exp_name Leaderboard_pred -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt -sam_config sam2_hiera_t -pretrain logs/BTCV_MedSAM2_2024_10_26_18_40_21/Model/latest_epoch.pth -image_size 1024 -out_size 512 -val_freq 1 -prompt bbox -prompt_freq 1 -dataset leaderboard -data_path ./data/Public_leaderboard_data
# srun zip -r output/SAM2_s.zip output/test_labels