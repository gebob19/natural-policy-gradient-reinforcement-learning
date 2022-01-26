optim_name=$1
hparam_path=visualize/json/baseline.json

run () {
    env_name=$1
    python main.py --rgb_input False --async_env False --optim_name $optim_name \
        --num_steps 1000 --global_n_steps 5e6 \
        --async_env False --writer_path runs/5e6_baseline/$env_name/ \
        --env_name $env_name --save_weights False \
        --hparam_path $hparam_path
}

# 5 seeds -- 2 in parralel
multiseed () {
    env_name=$1
    for i in 1..2
    do
        run $env_name & run $env_name
    done 
    run $env_name
}

# ~30 min per two seeds 
# 1.5hr per 5 seeds 
# 7 envs * 1.5hr = 10.5 hrs 
environments=(HalfCheetah-v2 Ant-v2 Hopper-v2 Humanoid-v2 Walker2d-v2 Reacher-v2 Swimmer-v2)
for env_name in ${environments[@]}; do
    multiseed $env_name
done

