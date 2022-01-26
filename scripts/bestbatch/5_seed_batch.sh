#!/bin/bash
#SBATCH --time=1-15:00:00
#SBATCH --nodes=1 
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --account=def-mjshafie
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=15

optim_name=$1
hparam_path=/home/bgebotys/projects/def-mjshafie/bgebotys/ngrad_rl/visualize/best_batch_hparams.json

run () {
    env_name=$1
    python main.py --rgb_input False --async_env False --optim_name $optim_name \
        --num_steps 1000 --global_n_steps 5e6 \
        --async_env False --writer_path runs/5e6_best_batch/$env_name/ \
        --env_name $env_name --save_weights False \
        --hparam_path $hparam_path
}

# 5 seeds -- 2 in parralel
multiseed () {
    env_name=$1
    for i in 1 2; do
        run $env_name & run $env_name
    done 
    run $env_name
}

source /home/bgebotys/projects/def-mjshafie/bgebotys/TSM_thesis_torch/venv/bin/activate 

# ~30 min per two seeds 
# 1.5hr per 5 seeds 
# 7 envs * 1.5hr = 10.5 hrs 
environments=(HalfCheetah-v2 Ant-v2 Hopper-v2 Humanoid-v2 Walker2d-v2 Reacher-v2 Swimmer-v2)
for env_name in ${environments[@]}; do
    multiseed $env_name
done

