import gym
import math
import json
import torch
import pathlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def sample(batch, p):
    sampled_batch_len = int(batch.shape[0] * p)
    idxs = np.random.choice(batch.shape[0], size=sampled_batch_len, replace=False)
    sampled_batch = batch[idxs]
    return sampled_batch, idxs


def get_stepsize(policy, max_kl):
    grads = torch.cat([p.og_grad.flatten() for p in policy.parameters()])
    ngrads = torch.cat([p.grad.flatten() for p in policy.parameters()])
    alpha = torch.sqrt((2 * max_kl) / ((grads @ ngrads) + 1e-8))
    return alpha


def zero_grad(model, set_none=False):
    for p in model.parameters():
        if set_none:
            p.og_grad = None
            p.grad = None
        else:
            p.og_grad = torch.zeros_like(p)
            p.grad = torch.zeros_like(p)


def sgd_step(model, step_size):
    for p in model.parameters():
        p.data = p.data - step_size * p.grad.data


def orthog_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def set_seeds(experiments):
    for exp in experiments:
        exp.seed = np.random.randint(1e5)


def setup_writer(base_path, name, hparams):

    bpath = pathlib.Path(f"{base_path}/")
    bpath.mkdir(exist_ok=True, parents=True)

    writer_path = f"{base_path}/{name}"
    writer = SummaryWriter(writer_path)  # metrics
    print("saving to:", writer_path, "...")

    # save hyperparams for run
    dfile = open(f"{writer_path}/hparams.json", "w")
    dfile.write(json.dumps(hparams, indent=4, sort_keys=True))
    dfile.close()

    return writer, writer_path


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


n1_vec = lambda x: x.flatten()[:, None]

vec = lambda xs: torch.cat([x.reshape(-1) for x in xs])


def set_flat_grad_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        prev_ind += flat_size


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


def get_flat_grad_from(net):
    fgrads = torch.cat([param.grad.view(-1) for param in net.parameters()])
    return fgrads


def vector_to_parameter_list(vec, parameters):
    params_new = []
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param_new = vec[pointer : pointer + num_param].view_as(param).data
        params_new.append(param_new)
        pointer += num_param
    return list(params_new)


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
import numpy as onp


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = onp.zeros(shape)
        self._S = onp.zeros(shape)

    def push(self, x):
        x = onp.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else onp.square(self._M)

    @property
    def std(self):
        return onp.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = onp.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class RunningStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip=5):
        super().__init__(env)
        self.running_state = ZFilter(env.observation_space.shape, clip=clip)

    def observation(self, obs):
        return self.running_state(obs)


class TorchGym(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).squeeze().double()
        return obs

    def step(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        obs2, r, d, _ = self.env.step(action.squeeze())
        d = torch.tensor(d).double()
        r = torch.tensor(r).double()
        obs2 = torch.from_numpy(obs2).squeeze().double()
        return obs2, r, d, _


# dict -> namespace
class DictNamespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
