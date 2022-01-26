#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1 
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --account=def-mjshafie
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=15

env_name=HalfCheetah-v2
global_n_steps=5e6
hparam_path=/home/bgebotys/projects/def-mjshafie/bgebotys/ngrad_rl/visualize/batch_hparams.json

run () {
    num_envs=$1
    optim_name=$2
    
    # echo $num_envs $optim_name

    python main.py --rgb_input False --async_env False --optim_name $optim_name \
        --num_steps 1000 --global_n_steps $global_n_steps \
        --async_env False --writer_path runs/5e6_hparams_batch_runs/ \
        --env_name $env_name --save_weights False \
        --num_envs $num_envs \
        --hparam_path $hparam_path
}

## activate conda env 
source /home/bgebotys/projects/def-mjshafie/bgebotys/TSM_thesis_torch/venv/bin/activate 

# ~30 min per run
# * 5 different batch sizes * 5 different optims = 25 * 30 = 750 / 60minperhour = 12.5hours

optims=(diagonal kfac ekfac tengradv2 hessianfree)

for optim_name in ${optims[@]}; do

    if [ $optim_name == "tengradv2" ]; then 
        num_envs_search=(10 8 6 4)
    else
        num_envs_search=(25 20 15 10 5)
    fi

    for n_env in ${num_envs_search[@]}; do
        run $n_env $optim_name
    done
done