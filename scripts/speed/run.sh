## Note: 
## training scripts use multi-processing which can lead to 
## incorrect speed metrics -- so we have to run it in a 
## serial way -- which is why we use these scripts/speed/... scripts

writer_prefix=$1
hparam_path=$2
n_global_steps=100000

run () {
    env_name=$1
    optim_name=$2
    python main.py --rgb_input False --async_env False --optim_name $optim_name \
        --num_steps 1000 --global_n_steps $n_global_steps \
        --async_env False --writer_path runs/${writer_prefix}_speed/$env_name/ \
        --env_name $env_name --save_weights False \
        --hparam_path $hparam_path
}

optims=(diagonal kfac ekfac hessianfree tengradv2)
environments=(Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 Walker2d-v2 Reacher-v2 Swimmer-v2)

for optim_name in ${optims[@]}; do
    for env_name in ${environments[@]}; do
        run $env_name $optim_name
    done
done