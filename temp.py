

#ssh jwang3180@login-ice.pace.gatech.edu ssh to ICE

# activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate jazzmus
# WANDB_DISABLED=true  python fp-train-1.py --config_path config/FP-GrandStaff/pretraining.json

# python fp-train-1.py --config_path config/FP-GrandStaff/pretraining.json
# python fp-train-2.py --config_path config/FP-Polish_Scores/pretraining.json --starting_checkpoint /home/hice1/jwang3180/SMT/weights/GrandStaff/FP-Grandstaff-system-level-v1.ckpt


# 
# checking GPU and requesting GPU resources
# srun --mem=64G --gres=gpu:H200:1 --time=4:00:00 --pty bash
# srun --mem=64G --gres=gpu:H200:8 --time=4:00:00 --pty bash
# srun --mem=1900G --gres=gpu:H200:8 --cpus-per-task=64 --time=4:00:00 --pty bash


dataset = datasets.load_dataset("PRAIG/fp-grandstaff")
sample = dataset['test'][0]
print("GT format:", repr(sample['transcription'][:200]))
print("\nContains <b>?", '<b>' in sample['transcription'])
print("Contains \\n?", '\n' in sample['transcription'])

# scp -i ~/.ssh/lambda-key.pem ubuntu@192.222.58.136:/lambda/nfs/SMTTraining/SMT/requirements.txt ~/Downloads/
# ssh -i ~/.ssh/lambda-key.pem ubuntu@192.222.58.215
# go back to batch if the data the model blew up

srun -p ice-gpu --gres=gpu:a100:1 --mem=64G --cpus-per-task=20 --time=5:00:00 --mail-type=BEGIN --mail-user=jwang3180@gatech.edu --pty bash
srun -p ice-gpu --gres=gpu:1 \
  --cpus-per-task=32 --mem=64G \
  --time=08:00:00 --pty bash
scontrol show job 3310074 | grep -E "StartTime|Priority|Reason|SchedNodeList"
slurm_load_jobs error: Invalid job id specified
 
 sinfo -p ice-gpu -N -o "%20N %10P %5c %10m %15G %8t %10e %12C %12O"
 
 sinfo -p PACE-gpu -N -o "%20N %10P %5c %10m %15G %8t %10e %12C %12O"
 
 scp jwang3180@login-ice-1.pace.gatech.edu:/home/hice1/jwang3180/SMT/weights/GrandStaff/FP-Grandstaff-system-level-v1.ckpt /Users/jameswang/workspace/OMR-Jazz/SMT
 
 scp -i ~/.ssh/lambda-key.pem ~/Downloads/FP-Grandstaff-system-level-v1.ckpt ubuntu@192.222.58.35:/home/ubuntu/SMTTraining/SMT/weights/GrandStaff/

python train.py \
    --fold 0 \
    --config "config/replication/pretrained_syn_word.gin" \
    --model_type "smt" \
    --batch_size 64 \
    --accumulate_grad_batches 1 \
    --lr 5e-4 \
    --epochs 100 \
    --patience 10
    
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python train_full_page.py --config "config/full-page/full_page_syn_character.gin" --checkpoint_path "weights/smt_sys_best/smt_pre_syn_character.ckpt"
# SSH to ICE login node
ssh jwang3180@login-ice.pace.gatech.edu

# Activate conda environment (PACE Anaconda)
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate jazzmus

# Standard H200 GPU session (interactive shell)
srun -p ice-gpu --gres=gpu:h100:1 --mem=64G --cpus-per-task=20 \
  --time=4:00:00 --mail-type=BEGIN --mail-user=jwang3180@gatech.edu --pty bash
  
srun -p ice-gpu --gres=gpu:h200:1 --mem=64G --cpus-per-task=20 \
  --time=4:00:00 --mail-type=BEGIN --mail-user=jwang3180@gatech.edu --pty bash

srun -p ice-gpu --gres=gpu:v100:1 --mem=64G --cpus-per-task=10 \
  --time=6:00:00 --mail-type=BEGIN --mail-user=jwang3180@gatech.edu --pty bash

# 8x H200, very high memory/CPU (interactive)
srun --mem=1900G --gres=gpu:H200:8 --cpus-per-task=64 --time=4:00:00 --pty bash

# Basic GPU session (generic)
srun -p ice-gpu --gres=gpu:1 --cpus-per-task=32 --mem=64G \
  --time=08:00:00 --pty bash

#!/bin/bash
#SBATCH -J smt-train
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jwang3180@gatech.edu
#SBATCH -o logs/%x-%j.out

# Load conda and activate env
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate jazzmus

# Diagnostics
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Your training command
python train.py --config config/fosca_config_pretrained_syn.gin --epochs 100

# Submit / cancel / requeue
sbatch run_gpu.sbatch
scancel <job_id>
scontrol requeue <job_id>

# Queue overview (your jobs)
squeue -u $USER

# Only pending jobs with reasons
squeue -u $USER -t PD -o "%.18i %.9P %.8j %.8T %.10M %.6D %R"

# Job details (start time, priority, reason, candidate nodes)
scontrol show job <job_id> | grep -E "JobId=|StartTime|Priority|Reason|SchedNodeList"

# Node status (ICE / PACE partitions)
sinfo -p ice-gpu -N -o "%20N %10P %5c %10m %15G %8t %10e %12C %12O"
sinfo -p PACE-gpu -N -o "%20N %10P %5c %10m %15G %8t %10e %12C %12O"

# Completed/active job accounting (efficiency summary)
sacct -j <job_id> --format=JobID,JobName,Partition,AllocCPUS,AllocTRES,Elapsed,State,MaxRSS,MaxVMSize,ReqMem,NodeList

# SLURM job efficiency (if available)
seff <job_id>

# Remote (PACE) → Local
scp jwang3180@login-ice-1.pace.gatech.edu:/home/hice1/jwang3180/SMT/weights/GrandStaff/FP-Grandstaff-system-level-v1.ckpt \
    /Users/jameswang/workspace/OMR-Jazz/SMT

# SSH into Lambda GPU instance
ssh -i ~/.ssh/lambda-key.pem ubuntu@192.222.58.215

# SCP Local → Lambda
scp -i ~/.ssh/lambda-key.pem ~/Downloads/FP-Grandstaff-system-level-v1.ckpt \
    ubuntu@192.222.58.35:/home/ubuntu/SMTTraining/SMT/weights/GrandStaff/


# Training runs
python fp-train-1.py --config_path config/FP-GrandStaff/pretraining.json

python fp-train-2.py --config_path config/FP-Polish_Scores/pretraining.json \
  --starting_checkpoint /home/hice1/jwang3180/SMT/weights/GrandStaff/FP-Grandstaff-system-level-v1.ckpt

# Gin-config example
python train.py \
  --fold 0 \
  --config "config/full-page/fp_pre_.gin" \
  --model_type "smt" \
  --batch_size 4 \
  --accumulate_grad_batches 1 \
  --lr 1.25e-4 \
  --epochs 100 \
  --patience 10


# CUDA / GPU visibility
nvidia-smi
watch -n 2 nvidia-smi

# Torch reports
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"


# System / CPU / Memory / Disk
hostnamectl
lscpu
free -h
vmstat 1
iostat -xz 1      # if sysstat installed
df -h
du -sh ./* 2>/dev/null | sort -h

# Processes
top
htop               # if available
ps -u $USER -o pid,ppid,pcpu,pmem,stime,time,cmd --sort=-pcpu | head -n 20

# Network / ports
ip a
ss -tulpen
ping -c 4 login-ice.pace.gatech.edu
nc -zv host port   # connectivity check

# Environment
env | sort
which python
conda env list
conda info

# Identity & config
git config --global user.name "Jiayi Wang"
git config --global user.email "jwang3180@gatech.edu"
git config --global pull.rebase false

# Basic workflow
git status
git add -A
git commit -m "Update training configs"
git pull
git push

# Branching
git checkout -b exp/h200-tuning
git switch main
git merge exp/h200-tuning

# Navigation
pwd
ls -lah
cd /path/to/dir
mkdir -p logs checkpoints

# Copy / move / remove (CAREFUL with -r)
cp -r src/ backup/src_$(date +%F)
mv oldname newname
rm -rf tmp/*

# File inspection
head -n 50 file.txt
tail -n 50 file.txt
tail -f logs/train.out
less -S large.txt

# Search (grep / find)
grep -Rin --line-number "keyword" .
find . -type f -name "*.ckpt"
find . -type f -size +1G -print

# Disk usage
du -sh .
du -sh * | sort -h

# Links / permissions
ln -s /path/to/target linkname
chmod u+x script.sh
chmod -R g+rX some_dir
chown -R $USER:$USER some_dir   # if permitted

# Archive / extract
tar -czf archive.tgz some_dir
tar -xzf archive.tgz -C /dest

# Checksums
md5sum file
sha256sum file

# Date / time
date
TZ=UTC date
