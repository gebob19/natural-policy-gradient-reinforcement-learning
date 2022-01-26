#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1 
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --account=def-mjshafie
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=15

optim_name=$1
env_name=HalfCheetah-v2
global_n_steps=5e6

run () {
    value_update=$1
    value_damping=$2
    value_linesearch=$3
    value_lr_max=$4
    
    python main.py --rgb_input False --async_env False --optim_name $optim_name \
        --num_steps 1000 --global_n_steps $global_n_steps \
        --async_env False --writer_path runs/5e6_hparams_critic_runs/ \
        --env_name $env_name --save_weights False \
        --value_update $value_update --value_lr_max $value_lr_max \
        --value_damping $value_damping --value_linesearch $value_linesearch \
        --hparam_path /home/bgebotys/projects/def-mjshafie/bgebotys/ngrad_rl/visualize/best_post_batchruns.json
}

source /home/bgebotys/projects/def-mjshafie/bgebotys/TSM_thesis_torch/venv/bin/activate 

# ~30 min per run
# 5 * 2 * 2 = 20 * 30 = 600 / 60minperhour = 10hours
critic_optims=(tengradv2 diagonal kfac ekfac hessianfree)
dps=(1e-1 1e-2)
max_lrs=(1e-1 1e-2)

for v_optim_name in ${critic_optims[@]}; do
    for dp in ${dps[@]}; do
        # linesearch 
        run $v_optim_name $dp True 0
        # no linesearch
        for max_lr in ${max_lrs[@]}; do
            run $v_optim_name $dp False $max_lr
        done
    done
done
