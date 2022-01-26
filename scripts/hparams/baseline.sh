#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1 
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --account=def-mjshafie
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=15

global_n_steps=1e6
env_name=HalfCheetah-v2
writer_path=runs/hparams_baseline/

run () {
    optim_name=$1
    num_envs=$2
    damping=$3
    value_lr=$4 
    linesearch=$5
    lr_max=$6

    python main.py --optim_name $optim_name \
        --writer_path $writer_path --env_name $env_name \
        --rgb_input False --async_env False \
        --num_envs $num_envs --num_steps 1000 \
        --global_n_steps $global_n_steps \
        --linesearch $linesearch --damping $damping \
        --lr_max $lr_max
}

hparam () {
    optim_name=$1
    num_envs=$2

    # 2 * 2 * (1 + 2 = 3) = 4 * 6 = 24 runs 
    # ~5min per run * 24 runs = 2 hours for hparam tune
    # 5 different optims * 2 hours = 10hours 
    damping=(1e-1 1e-2)
    value_lr=(1e-2 1e-3)
    linesearchs=(True False)
    lr_max=(1e-1 1e-2)

    for damp in ${damping[@]}; do
        for v_lr in ${value_lr[@]}; do
            # with line search 
            run $optim_name $num_envs $damp $v_lr True 0
            
            # without line search 
            for lrMAX in ${lr_max[@]}; do
                run $optim_name $num_envs $damp $v_lr False $lrMAX
            done 
        done 
    done 
}

## run with default batchsizes -- see appendix for tengrad info
hparam diagonal 15 &&
hparam hessianfree 15 &&
hparam kfac 15 &&
hparam ekfac 15 &&
hparam tengradv2 6