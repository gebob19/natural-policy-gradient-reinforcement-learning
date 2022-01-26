#%%
import gym
import time
import json
import torch
import pprint
import argparse
import numpy as np
import distutils.util

from tqdm import tqdm

from optim import Diagonal, KFAC, EKFAC, NGDOptimizer
from optim import HessianFree, linesearch, DKL_continuous

from utils import DictNamespace
from utils import normal_log_density
from utils import setup_writer, set_seeds
from utils import TorchGym, RunningStateWrapper
from utils import get_stepsize, zero_grad, sgd_step
from utils import set_flat_params_to, vec, sample, set_flat_grad_to

from models import Policy, Value

torch.set_default_tensor_type("torch.DoubleTensor")
torch.utils.backcompat.keepdim_warning.enabled = True
torch.utils.backcompat.broadcast_warning.enabled = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# sshhhh -- should turn this off when debuging
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def multi_env_rollout(
    agents, envs, memory, args, device, global_step, next_obs, next_done
):
    (obs, actions, rewards, dones) = memory
    (policy_net, _) = agents
    eoe_rewards = []

    steps = range(0, args.num_steps)
    if not args.silent:
        steps = tqdm(steps)

    for step in steps:
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action_mean, _, action_std = policy_net(next_obs)
            action = torch.normal(action_mean, action_std)

        actions[step] = action

        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(
            device
        )

        for item in info:
            if "episode" in item.keys():
                eoe_rewards.append(item["episode"]["r"])

    return global_step, next_obs, next_done, eoe_rewards


def compute_advantage_targets(value_net, batch, gamma, tau):
    (states, actions, rewards, masks) = batch
    with torch.no_grad():
        values = value_net(states)

    returns = torch.Tensor(actions.size(0), 1).to(device)
    deltas = torch.Tensor(actions.size(0), 1).to(device)
    advantages = torch.Tensor(actions.size(0), 1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = returns
    advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, targets


def policy_loss(policy_net, batch):
    (states, actions, advantages, fixed_log_probs) = batch

    z = policy_net(states)
    log_prob = normal_log_density(actions, *z)
    ploss = -(advantages * torch.exp(log_prob - fixed_log_probs))
    return ploss.mean()


def value_loss(value_net, batch, l2_reg):
    (states, targets) = batch
    values_ = value_net(states)
    vloss = (values_ - targets).pow(2).mean()

    # weight decay
    for param in value_net.parameters():
        vloss += param.pow(2).sum() * l2_reg
    return vloss


def multi_env_flatten(data):
    # (nsteps, n_envs, *d) -> (nsteps * n_envs, *d)
    stack = torch.cat([data[:, i] for i in range(data.shape[1])], 0).squeeze(-1)
    return stack


def update_HF(args, preconditioner, network, loss_fcn, get_kl):
    assert preconditioner._name == "hessianfree"
    loss = loss_fcn()
    zero_grad(network)

    alpha, stepdir, neggdotstepdir = preconditioner.step(get_kl, loss)

    if args.value_linesearch:
        fullstep = alpha * stepdir
        expected_improve = alpha * neggdotstepdir

        with torch.no_grad():
            prev_params = vec(network.parameters())
            _, new_params = linesearch(
                network,
                loss_fcn,
                prev_params,
                fullstep,
                expected_improve,
                verbose=not args.silent,
            )
            set_flat_params_to(network, new_params)
    else:
        set_flat_grad_to(network, -stepdir)
        alpha = min(alpha, args.lr_max)
        sgd_step(network, alpha)


def update_parametric(
    args, preconditioner, network, loss_fcn, z, actions, value_update
):
    assert preconditioner._name in ["kfac", "ekfac", "diagonal", "tengradv2"]
    loss = loss_fcn()
    zero_grad(network)

    # collect the gradients and activations for the Fisher-information matrix
    mu, _, std = z
    preconditioner.start_acc_stats()  ## this starts the collection
    ## natural fisher update -- optimized sampled actions
    if args.natural_fisher or value_update:
        with torch.no_grad():
            sampled = torch.normal(mu, std)

        ### MSE or logprob (ACKTR used MSE in their implementation)
        if value_update and args.mse_value_fisher:
            fshr_log_prob = -((sampled - mu) ** 2).mean()
        else:
            fshr_log_prob = normal_log_density(sampled, *z).mean()
    ## empirical fisher update -- optimized the actions taken during rollout
    else:
        fshr_log_prob = normal_log_density(actions, *z).mean()
    fshr_log_prob.backward(retain_graph=True)
    preconditioner.stop_acc_stats()

    # compute actual gradients
    zero_grad(network)
    loss.backward()
    preconditioner.step()  # precondition with natural gradient i.e. = F^-1 grad

    alpha = get_stepsize(network, args.max_kl)

    # policy VS critic network update
    linesearch_flag = args.value_linesearch if value_update else args.linesearch
    lr_max = args.value_lr_max if value_update else args.lr_max

    if linesearch_flag:
        stepdir = -torch.cat([p.grad.flatten() for p in network.parameters()])
        fullstep = alpha * stepdir

        fgrads = torch.cat([p.og_grad.flatten() for p in network.parameters()])
        neggdotstepdir = (-fgrads * stepdir).sum(0, keepdim=True)
        expected_improve = alpha * neggdotstepdir

        with torch.no_grad():
            prev_params = vec(network.parameters())
            _, new_params = linesearch(
                network,
                loss_fcn,
                prev_params,
                fullstep,
                expected_improve,
                verbose=not args.silent,
            )
            set_flat_params_to(network, new_params)
    else:

        alpha = min(alpha, lr_max)
        sgd_step(network, alpha)


#%%
def train(args):
    if args.silent:
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    else:

        pprint.pprint(vars(args))

    device = torch.device(0)

    # setup Tensorboard metric writer
    exp_name = f"{args.seed}_{args.env_name}_{args.optim_name}"
    if not args.silent:
        print(f"training {exp_name}...")

    base_path = f"{args.writer_path}/{args.optim_name}_runs"
    writer, writer_path = setup_writer(base_path, exp_name, vars(args))

    # create multi-envs for rollout
    torch.manual_seed(args.seed)

    def make_env(env_name, seed):
        def thunk():
            env = gym.make(env_name)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = RunningStateWrapper(env)
            env = TorchGym(env)

            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    if args.async_env:
        multi_env_fcn = gym.vector.AsyncVectorEnv
    else:
        multi_env_fcn = gym.vector.SyncVectorEnv

    envs = multi_env_fcn(
        [make_env(args.env_name, args.seed + i) for i in range(args.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    if not args.silent:
        print(f"device: {device}")
        print(f"obs dim: {obs_dim} action_dim: {action_dim}")

    # memory setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    memory = (obs, actions, rewards, dones)

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # setup models
    policy_net = Policy(obs_dim, action_dim)
    value_net = Value(obs_dim)
    ## this is for a 3 GPU setup -- if you have less change to device
    vdevice = torch.device(1)  # device
    pdevice = torch.device(2)  # device

    if not args.silent:
        print(f"p_device: {pdevice} v_device: {vdevice}")

    policy_net = policy_net.to(device)
    value_net = value_net.to(device)

    # policy natural gradient optimizer
    if not args.silent:
        print(f"using second order method: {args.optim_name}")

    if args.optim_name == "hessianfree":
        preconditioner = HessianFree(
            policy_net, args.damping, args.max_kl, args.n_cg_steps
        )
    elif args.optim_name == "diagonal":
        preconditioner = Diagonal(policy_net, args.damping, alpha=args.momentum)
    elif args.optim_name == "kfac":
        preconditioner = KFAC(
            policy_net,
            args.damping,
            update_freq=args.TInv,
            pi=True,
            alpha=args.momentum,
        )
    elif args.optim_name == "ekfac":
        preconditioner = EKFAC(
            policy_net,
            args.damping,
            update_freq=args.TInv,
            ra=True,
            alpha=args.momentum,
        )

    elif args.optim_name == "tengradv2":
        preconditioner = NGDOptimizer(
            policy_net,
            momentum=args.momentum,
            damping=args.damping,
            weight_decay=0,
            freq=args.TInv,
            gamma=0.99,
            low_rank="false",
        )
    else:
        raise NotImplementedError

    # policy natural gradient optimizer
    if args.value_update == "sgd":
        voptim = torch.optim.SGD(
            value_net.parameters(), momentum=args.momentum, lr=args.value_lr
        )
    elif args.value_update == "diagonal":
        value_preconditioner = Diagonal(
            value_net, args.value_damping, alpha=args.momentum
        )
    elif args.value_update == "kfac":
        value_preconditioner = KFAC(
            value_net,
            args.value_damping,
            update_freq=args.TInv,
            pi=True,
            alpha=args.momentum,
        )
    elif args.value_update == "ekfac":
        value_preconditioner = EKFAC(
            value_net,
            args.value_damping,
            update_freq=args.TInv,
            ra=True,
            alpha=args.momentum,
        )
    elif args.value_update == "hessianfree":
        value_preconditioner = HessianFree(
            value_net, args.value_damping, args.max_kl, args.n_cg_steps
        )
    elif args.value_update == "tengradv2":
        value_preconditioner = NGDOptimizer(
            value_net,
            momentum=args.momentum,
            damping=args.value_damping,
            weight_decay=0,
            freq=args.TInv,
            gamma=0.99,
            low_rank="false",
        )
    else:
        raise NotImplementedError

    # training
    metrics = []
    poptim_times = []
    poptim_times = []
    n_updates = 0
    global_step = 0
    max_reward = -float("inf")

    pbar = tqdm(total=args.global_n_steps)

    while global_step < args.global_n_steps:
        start = time.time()

        # rollout in env
        policy_net = policy_net.to(device)
        value_net = value_net.to(device)

        global_step, next_obs, next_done, eoe_rewards = multi_env_rollout(
            (policy_net, value_net),
            envs,
            memory,
            args,
            device,
            global_step,
            next_obs,
            next_done,
        )

        # post-processing advantage & GAE
        fmemory = [multi_env_flatten(m) for m in memory]
        fmemory[-1] = 1 - fmemory[-1]  # dones -> mask
        states, actions, *_ = fmemory
        advantages, targets = compute_advantage_targets(
            value_net, fmemory, args.gamma, args.tau
        )

        # metric logging
        if len(eoe_rewards) > 0:
            avg_reward = np.mean(eoe_rewards)
            if not args.silent:
                print(
                    "Update {}\t Step {}\t Average reward {:.2f}".format(
                        n_updates, global_step, avg_reward
                    )
                )

            writer.add_scalar("reward/mean", avg_reward, global_step)

            m = [float(m) for m in [avg_reward, time.time(), n_updates, global_step]]
            metrics.append(m)

            ## save best models
            if args.save_weights and avg_reward > max_reward:
                if not args.silent:
                    print(f"saving weights to {writer_path}...")
                max_reward = avg_reward
                torch.save(policy_net.state_dict(), f"{writer_path}/policy_net_best.sd")
                torch.save(value_net.state_dict(), f"{writer_path}/value_net_best.sd")

        n_updates += 1

        # optimize on different gpus (bc OOM otherwise)
        value_net = value_net.to(vdevice)
        policy_net = policy_net.to(pdevice)

        # update critic network
        states, targets = states.to(vdevice), targets.to(vdevice)
        if args.value_update == "sgd":
            voptim.zero_grad()
            vloss = value_loss(value_net, (states, targets), args.l2_reg)
            vloss.backward()
            voptim.step()

        elif value_preconditioner._name == "hessianfree":

            def get_kl():
                values = value_net(states)
                vstd = torch.ones_like(values).to(vdevice)  # static ones
                log_vstd = torch.log(vstd)

                # mu0, log_std0, std0 = z0
                z = (values, log_vstd, vstd)
                z1 = tuple([zi.data for zi in z])

                return DKL_continuous(z1, z)

            loss_fcn = lambda: value_loss(value_net, (states, targets), args.l2_reg)
            update_HF(args, value_preconditioner, value_net, loss_fcn, get_kl)

        elif value_preconditioner._name in ["kfac", "ekfac", "diagonal", "tengradv2"]:
            vstates, vtargets = states, targets
            if value_preconditioner._name == "tengradv2" and vstates.shape[0] > 4000:
                # subsample to batchsize of 4000 (best batchsize) -- OOM otherwise
                # bc of this the performance with tengrad for critic optim was never good.
                # you can remove this if you have better hardware.
                idxs = np.random.randint(0, high=len(states), size=(4000,))
                vstates = states[idxs]
                vtargets = targets[idxs]

            get_loss = lambda: value_loss(value_net, (vstates, vtargets), args.l2_reg)

            values = value_net(vstates)
            vstd = torch.ones_like(values).to(vdevice)  # static ones
            log_vstd = torch.log(vstd)
            z = (values, log_vstd, vstd)
            update_parametric(
                args,
                value_preconditioner,
                value_net,
                get_loss,
                z,
                actions,
                value_update=True,
            )

        # optimize the policy
        poptim_time = time.time()
        states, actions = states.to(pdevice), actions.to(pdevice)
        z = policy_net(states)  # z = (mu, log_vstd, std) for actions
        fixed_log_probs = normal_log_density(actions, *z).data.clone()
        batch = (states, actions, advantages, fixed_log_probs)
        batch = [b.to(pdevice) for b in batch]

        get_loss = lambda: policy_loss(policy_net, batch)

        if preconditioner._name == "hessianfree":
            ## sample update like in TRPO
            batch_states = batch[0]
            if args.sample_size < 1:
                batch_states = sample(batch_states, args.sample_size)[0]

            def get_kl():
                z = policy_net(batch_states)
                z1 = tuple([zi.data for zi in z])
                return DKL_continuous(z1, z)

            update_HF(args, preconditioner, policy_net, get_loss, get_kl)

        elif preconditioner._name in ["kfac", "ekfac", "diagonal", "tengradv2"]:
            update_parametric(
                args,
                preconditioner,
                policy_net,
                get_loss,
                z,
                actions,
                value_update=False,
            )
        else:
            raise NotImplementedError

        # collect more metrics
        poptim_times.append(time.time() - poptim_time)
        if not args.silent:
            print(f"{time.time() - start :.2f} sec per grad step")

        pbar.update(args.num_envs * args.num_steps)

    pbar.close()

    print("----------------")
    print(
        f"{preconditioner._name.upper()}: {np.mean(poptim_times) :.5f} sec per policy optim step"
    )
    print("----------------")
    writer.close()

    dfile = open(f"{writer_path}/time.json", "w")
    dfile.write(json.dumps({"mean_poptim_time": f"{np.mean(poptim_times) :.5f}"}))
    dfile.close()

    dfile = open(f"{writer_path}/results.json", "w")
    dfile.write(json.dumps(metrics))
    dfile.close()


def get_default_args():
    ## defaults for all the parse args
    ## can also use this for programatic runs however, had pytorch cuda
    ## memory doesn't clear nicely after experiments in a for loop.
    ## usually its better to just program in shell.
    args = {}
    args["seed"] = 0
    args["gamma"] = 0.995
    args["tau"] = 0.97
    args["l2_reg"] = 0.001
    args["use_running_state"] = True
    args["silent"] = True  # False

    ### second order methods
    args["optim_name"] = "hessianfree"
    args["damping"] = 0.1
    args["linesearch"] = True
    args["lr_max"] = 1e-3  ## step size clipping
    # hessianfree
    args["max_kl"] = 0.01
    args["n_cg_steps"] = 10
    args["sample_size"] = 0.1
    # kfac/ekfac/diag/tengrad
    args["natural_fisher"] = False
    args["mse_value_fisher"] = True
    args["momentum"] = 0.95
    # kfac/ekfac/tengrad
    args["TInv"] = 20

    # critic optimizaiton
    args["value_update"] = "sgd"  # 'diagonal', 'sgd', 'kfac', etc.
    args["value_lr_max"] = 1e-3
    args["value_lr"] = 1e-3
    args["value_damping"] = 1e-2
    args["value_linesearch"] = True

    # hparam experiments
    args["env_name"] = "HalfCheetah-v2"  ## max steps = 1000
    args["save_weights"] = False  # !!
    args["global_n_steps"] = 1e6
    # batch size = num_envs * num_steps
    args["num_steps"] = 1000
    args["num_envs"] = 10

    args["writer_path"] = "{}_runs/".format(
        args["env_name"]
    )  # folder to save all files in
    args["rgb_input"] = False  ## !!
    args["async_env"] = True if args["rgb_input"] else False

    return args


def shell_run():
    args = get_default_args()  # these will be the default values
    parser = argparse.ArgumentParser()

    def add_boolean_arg(name):
        parser.add_argument(
            f"--{name}",
            type=lambda x: bool(distutils.util.strtobool(x)),
            default=args[f"{name}"],
            required=False,
        )

    # change the default values by using flags
    # fmt: off
    parser.add_argument("--seed", type=int, default=args["seed"], required=False)
    parser.add_argument("--gamma", type=float, default=args["gamma"], required=False)
    parser.add_argument("--tau", type=float, default=args["tau"], required=False)
    parser.add_argument("--l2_reg", type=float, default=args["l2_reg"], required=False)
    add_boolean_arg("use_running_state")
    add_boolean_arg("async_env")
    add_boolean_arg("silent")
    parser.add_argument("--optim_name", type=str, default=args["optim_name"], required=False)
    parser.add_argument("--damping", type=float, default=args["damping"], required=False)
    parser.add_argument("--lr_max", type=float, default=args["lr_max"], required=False)
    add_boolean_arg("linesearch")

    parser.add_argument("--max_kl", type=float, default=args["max_kl"], required=False)
    parser.add_argument("--n_cg_steps", type=int, default=args["n_cg_steps"], required=False)
    parser.add_argument("--sample_size", type=float, default=args["sample_size"], required=False)
    parser.add_argument("--momentum", type=float, default=args["momentum"], required=False)
    parser.add_argument("--TInv", type=float, default=args["TInv"], required=False)
    add_boolean_arg("natural_fisher")
    add_boolean_arg("mse_value_fisher")

    parser.add_argument("--value_update", type=str, default=args["value_update"], required=False)
    parser.add_argument("--value_lr_max", type=float, default=args["value_lr_max"], required=False)
    parser.add_argument("--value_lr", type=float, default=args["value_lr"], required=False)
    parser.add_argument("--value_damping", type=float, default=args["value_damping"], required=False)
    add_boolean_arg("value_linesearch")

    parser.add_argument("--env_name", type=str, default=args["env_name"], required=False)
    parser.add_argument("--global_n_steps", type=float, default=args["global_n_steps"], required=False)
    add_boolean_arg("save_weights")

    parser.add_argument("--num_steps", type=int, default=args["num_steps"], required=False)
    parser.add_argument("--num_envs", type=int, default=args["num_envs"], required=False)
    parser.add_argument("--writer_path", type=str, default=args["writer_path"], required=False)
    add_boolean_arg("rgb_input")

    # load hyperparmameters from a path 
    # and overwrite all the hyperparameters with the hparams specified in the 
    # file 
    parser.add_argument(f"--hparam_path", type=str, default="", required=False)
    # fmt: on

    args = parser.parse_args()

    if args.hparam_path != "":
        # load hyperparameter file
        with open(args.hparam_path, "r") as f:
            hparams = json.load(f)
        hparams = hparams[args.optim_name]

        # update argument values with tuned values
        args = vars(args)
        for hp in hparams:
            if hp in args:
                t = type(args[hp])
                args[hp] = t(hparams[hp])
            else:
                print(f"Warning: skipping {hp} hparameter...")
        args = DictNamespace(args)

    if args.seed == 0:  # set a random seed
        set_seeds([args])

    pprint.pprint(vars(args))

    train(args)


if __name__ == "__main__":
    shell_run()
