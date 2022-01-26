# Built off of https://github.com/Thrandis/EKFAC-pytorch (EKFAC + KFAC impls)
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Unfold
from torch.linalg import inv, svd
from torch.optim.optimizer import Optimizer
from torch import einsum, eye, matmul, cumsum
from utils import vec, set_flat_params_to, get_flat_grad_from

class NGDOptimizer(optim.Optimizer):
    def __init__(
        self,
        model,
        lr=0.01,
        momentum=0.9,
        damping=0.1,
        kl_clip=0.01,
        weight_decay=0.003,
        freq=100,
        gamma=0.9,
        low_rank="true",
        super_opt="false",
        reduce_sum="false",
        diag="false",
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay
        )

        super(NGDOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {"Linear", "Conv2d"}
        self.modules = []
        # self.grad_outputs = {}
        self.IHandler = ComputeI()
        self.GHandler = ComputeG()
        self.model = model
        self._prepare_model()

        self._name = "tengradv2"
        self.steps = 0
        self.m_I = {}
        self.m_G = {}
        self.m_UV = {}
        self.m_NGD_Kernel = {}
        self.m_bias_Kernel = {}

        self.kl_clip = kl_clip
        self.freq = freq
        self.gamma = gamma
        self.low_rank = low_rank
        self.super_opt = super_opt
        self.reduce_sum = reduce_sum
        self.diag = diag
        self.damping = damping

    def _save_input(self, module, input):
        # storing the optimized input in forward pass
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            II, I = self.IHandler(
                input[0].data, module, self.super_opt, self.reduce_sum, self.diag
            )
            self.m_I[module] = II, I

    def _save_grad_output(self, module, grad_input, grad_output):
        # storing the optimized gradients in backward pass
        if self.acc_stats and self.steps % self.freq == 0:
            GG, G = self.GHandler(
                grad_output[0].data, module, self.super_opt, self.reduce_sum, self.diag
            )
            self.m_G[module] = GG, G

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("NGD keeps the following modules:")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print("(%s): %s" % (count, module))
                count += 1

    def _update_inv(self, m):
        classname = m.__class__.__name__.lower()
        if classname == "linear":
            assert m.optimized == True
            II = self.m_I[m][0]
            GG = self.m_G[m][0]
            n = II.shape[0]

            ### bias kernel is GG (II = all ones)
            bias_kernel = GG / n
            bias_inv = inv(bias_kernel + self.damping * eye(n).to(GG.device))
            self.m_bias_Kernel[m] = bias_inv

            NGD_kernel = (II * GG) / n
            NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
            self.m_NGD_Kernel[m] = NGD_inv

            self.m_I[m] = (None, self.m_I[m][1])
            self.m_G[m] = (None, self.m_G[m][1])
            torch.cuda.empty_cache()
        elif classname == "conv2d":
            # SAEED: @TODO: we don't need II and GG after computations, clear the memory
            if m.optimized == True:
                # print('=== optimized ===')
                II = self.m_I[m][0]
                GG = self.m_G[m][0]
                n = II.shape[0]

                NGD_kernel = None
                if self.reduce_sum == "true":
                    if self.diag == "true":
                        NGD_kernel = II * GG / n
                        NGD_inv = torch.reciprocal(NGD_kernel + self.damping)
                    else:
                        NGD_kernel = II * GG / n
                        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
                else:
                    NGD_kernel = (einsum("nqlp->nq", II * GG)) / n
                    NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))

                self.m_NGD_Kernel[m] = NGD_inv

                self.m_I[m] = (None, self.m_I[m][1])
                self.m_G[m] = (None, self.m_G[m][1])
                torch.cuda.empty_cache()
            else:
                # SAEED: @TODO memory cleanup
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                AX = einsum("nkl,nml->nkm", (I, G))

                del I
                del G

                AX_ = AX.reshape(n, -1)
                out = matmul(AX_, AX_.t())

                del AX

                NGD_kernel = out / n
                ### low-rank approximation of Jacobian
                if self.low_rank == "true":
                    # print('=== low rank ===')
                    V, S, U = svd(AX_.T, full_matrices=False)
                    U = U.t()
                    V = V.t()
                    cs = cumsum(S, dim=0)
                    sum_s = sum(S)
                    index = ((cs - self.gamma * sum_s) <= 0).sum()
                    U = U[:, 0:index]
                    S = S[0:index]
                    V = V[0:index, :]
                    self.m_UV[m] = U, S, V

                del AX_

                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(NGD_kernel.device))
                self.m_NGD_Kernel[m] = NGD_inv

                del NGD_inv
                self.m_I[m] = None, self.m_I[m][1]
                self.m_G[m] = None, self.m_G[m][1]
                torch.cuda.empty_cache()

    def _get_natural_grad(self, m, damping):
        grad = m.weight.grad.data
        classname = m.__class__.__name__.lower()

        m.weight.og_grad = grad
        if m.bias is not None:
            m.bias.og_grad = m.bias.grad.data

        if classname == "linear":
            assert m.optimized == True
            I = self.m_I[m][1]
            G = self.m_G[m][1]
            n = I.shape[0]
            NGD_inv = self.m_NGD_Kernel[m]
            grad_prod = einsum("ni,oi->no", (I, grad))
            grad_prod = einsum("no,no->n", (grad_prod, G))

            v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

            gv = einsum("n,no->no", (v, G))
            gv = einsum("no,ni->oi", (gv, I))
            gv = gv / n

            bias_update = None
            if m.bias is not None:
                grad_bias = m.bias.grad.data
                if self.steps % self.freq == 0:
                    grad_prod_bias = einsum("o,no->n", (grad_bias, G))
                    v = matmul(
                        self.m_bias_Kernel[m], grad_prod_bias.unsqueeze(1)
                    ).squeeze()
                    gv_bias = einsum("n,no->o", (v, G))
                    gv_bias = gv_bias / n
                    bias_update = (grad_bias - gv_bias) / damping
                else:
                    bias_update = grad_bias

            updates = (grad - gv) / damping, bias_update

        elif classname == "conv2d":
            grad_reshape = grad.reshape(grad.shape[0], -1)
            if m.optimized == True:
                # print('=== optimized ===')
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                NGD_inv = self.m_NGD_Kernel[m]

                if self.reduce_sum == "true":
                    x1 = einsum("nk,mk->nm", (I, grad_reshape))
                    grad_prod = einsum("nm,nm->n", (x1, G))

                    if self.diag == "true":
                        v = NGD_inv * grad_prod
                    else:
                        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                    gv = einsum("n,nm->nm", (v, G))
                    gv = einsum("nm,nk->mk", (gv, I))
                else:
                    x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                    grad_prod = einsum("nml,nml->n", (x1, G))
                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                    gv = einsum("n,nml->nml", (v, G))
                    gv = einsum("nml,nkl->mk", (gv, I))
                gv = gv.view_as(grad)
                gv = gv / n

                bias_update = None
                if m.bias is not None:
                    bias_update = m.bias.grad.data

                updates = (grad - gv) / damping, bias_update

            else:
                # TODO(bmu): fix low rank
                if self.low_rank.lower() == "true":
                    # print("=== low rank ===")

                    ###### using low rank structure
                    U, S, V = self.m_UV[m]
                    NGD_inv = self.m_NGD_Kernel[m]
                    G = self.m_G[m][1]
                    n = NGD_inv.shape[0]

                    grad_prod = V @ grad_reshape.t().reshape(-1, 1)
                    grad_prod = torch.diag(S) @ grad_prod
                    grad_prod = U @ grad_prod
                    grad_prod = grad_prod.squeeze()

                    bias_update = None
                    if m.bias is not None:
                        bias_update = m.bias.grad.data

                    v = matmul(NGD_inv, (grad_prod).unsqueeze(1)).squeeze()

                    gv = U.t() @ v.unsqueeze(1)
                    gv = torch.diag(S) @ gv
                    gv = V.t() @ gv

                    gv = gv.reshape(grad_reshape.shape[1], grad_reshape.shape[0]).t()
                    gv = gv.view_as(grad)
                    gv = gv / n

                    updates = (grad - gv) / damping, bias_update
                else:
                    I = self.m_I[m][1]
                    G = self.m_G[m][1]
                    AX = einsum("nkl,nml->nkm", (I, G))

                    del I
                    del G

                    n = AX.shape[0]

                    NGD_inv = self.m_NGD_Kernel[m]

                    grad_prod = einsum("nkm,mk->n", (AX, grad_reshape))
                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                    gv = einsum("nkm,n->mk", (AX, v))
                    gv = gv.view_as(grad)
                    gv = gv / n

                    bias_update = None
                    if m.bias is not None:
                        bias_update = m.bias.grad.data

                    updates = (grad - gv) / damping, bias_update

                    del AX
                    del NGD_inv
                    torch.cuda.empty_cache()

        return updates

    def _kl_clip_and_update_grad(self, updates, lr):
        for m in self.model.modules():
            if m.__class__.__name__ in ["Linear", "Conv2d"]:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                # m.weight.grad.data.mul_(nu)
                if v[1] is not None:
                    m.bias.grad.data.copy_(v[1])
                    # m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                # print('=== step ===')
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # if weight_decay != 0 and self.steps >= 10 * self.freq:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group["lr"], d_p)
                # print('d_p:', d_p.shape)
                # print(d_p)

    def precondition(self, closure=None):
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.freq == 0:
                self._update_inv(m)
            v = self._get_natural_grad(m, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

    def step(self, closure=None):
        self.precondition(closure=closure)

        self._step(closure)
        self.steps += 1

    def start_acc_stats(self):
        self.acc_stats = True

    def stop_acc_stats(self):
        self.acc_stats = False


class ComputeI:
    @classmethod
    def compute_cov_a(
        cls, a, module, super_opt="false", reduce_sum="false", diag="false"
    ):
        return cls.__call__(a, module, super_opt, reduce_sum, diag)

    @classmethod
    def __call__(cls, a, module, super_opt="false", reduce_sum="false", diag="false"):
        if isinstance(module, nn.Linear):
            II, I = cls.linear(a, module, super_opt, reduce_sum, diag)
            return II, I
        elif isinstance(module, nn.Conv2d):
            II, I = cls.conv2d(a, module, super_opt, reduce_sum, diag)
            return II, I
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            return None

    @staticmethod
    def conv2d(input, module, super_opt="false", reduce_sum="false", diag="false"):
        f = Unfold(
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        I = f(input)
        N = I.shape[0]
        K = I.shape[1]
        L = I.shape[2]
        M = module.out_channels
        module.param_shapes = [N, K, L, M]

        if reduce_sum == "true":
            I = einsum("nkl->nk", I)
            if diag == "true":
                I /= L
                II = torch.sum(I * I, dim=1)
            else:
                II = einsum("nk,qk->nq", (I, I))
            module.optimized = True
            return II, I

        flag = False
        if super_opt == "true":
            flag = N * (L * L) * (K + M) < K * M * L + N * K * M
        else:
            flag = (L * L) * (K + M) < K * M

        if flag == True:
            II = einsum("nkl,qkp->nqlp", (I, I))
            module.optimized = True
            return II, I
        else:
            module.optimized = False
            return None, I

    @staticmethod
    def linear(input, module, super_opt="false", reduce_sum="false", diag="false"):
        I = input
        II = einsum("ni,li->nl", (I, I))
        module.optimized = True
        return II, I


class ComputeG:
    @classmethod
    def compute_cov_g(
        cls, g, module, super_opt="false", reduce_sum="false", diag="false"
    ):
        """
        :param g: gradient
        :param module: the corresponding module
        :return:
        """
        return cls.__call__(g, module, super_opt, reduce_sum, diag)

    @classmethod
    def __call__(cls, g, module, super_opt="false", reduce_sum="false", diag="false"):
        if isinstance(module, nn.Conv2d):
            GG, G = cls.conv2d(g, module, super_opt, reduce_sum, diag)
            return GG, G
        elif isinstance(module, nn.Linear):
            GG, G = cls.linear(g, module, super_opt, reduce_sum, diag)
            return GG, G
        else:
            return None

    @staticmethod
    def conv2d(g, module, super_opt="false", reduce_sum="false", diag="false"):
        n = g.shape[0]
        g_out_sc = n * g
        grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
        G = grad_output_viewed

        N = module.param_shapes[0]
        K = module.param_shapes[1]
        L = module.param_shapes[2]
        M = module.param_shapes[3]

        if reduce_sum == "true":
            G = einsum("nkl->nk", G)
            if diag == "true":
                G /= L
                GG = torch.sum(G * G, dim=1)
            else:
                GG = einsum("nk,qk->nq", (G, G))
            module.optimized = True
            return GG, G

        flag = False
        if super_opt == "true":
            flag = N * (L * L) * (K + M) < K * M * L + N * K * M
        else:
            flag = (L * L) * (K + M) < K * M

        if flag == True:
            GG = einsum("nml,qmp->nqlp", (G, G))
            module.optimized = True
            return GG, G
        else:
            module.optimized = False
            return None, G

    @staticmethod
    def linear(g, module, super_opt="false", reduce_sum="false", diag="false"):
        n = g.shape[0]
        g_out_sc = n * g
        G = g_out_sc
        GG = einsum("no,lo->nl", (G, G))
        module.optimized = True
        return GG, G


def _get_gathering_filter(mod):
    """Convolution filter that extracts input patches."""
    kw, kh = mod.kernel_size
    g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
    g_filter.fill_(0)
    for i in range(mod.in_channels):
        for j in range(kw):
            for k in range(kh):
                g_filter[k + kh * j + kw * kh * i, 0, j, k] = 1
    return g_filter

class Diagonal(Optimizer):
    def __init__(self, net, eps, alpha):
        self._name = "diagonal"
        self.eps = eps
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.acc_stats = False

        if alpha == 0:
            print("==> Not using rolling average optim")
        assert alpha < 1.0 and alpha > 0.0, "Invalid alpha value (i.e not in [0, 1])."

        # setup hooks
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ["Linear", "Conv2d"]:
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)

                d = {"params": params, "mod": mod, "layer_type": mod_class}
                self.params.append(d)
        super(Diagonal, self).__init__(self.params, {})

    def start_acc_stats(self):
        self.acc_stats = True

    def stop_acc_stats(self):
        self.acc_stats = False
        # store gradients
        for group in self.param_groups:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None

            state = self.state[weight]

            g = weight.grad.data
            s = g.shape

            if group["layer_type"] == "Conv2d":
                g = g.contiguous().view(s[0], s[1] * s[2] * s[3])

            if bias is not None:
                gb = bias.grad.data
                g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)

            if self._iteration_counter == 0:  # init
                state["gg"] = g ** 2
            else:
                state["gg"].mul_(self.alpha).add_(1 - self.alpha, g ** 2)

    def step(self):
        """Performs one step of preconditioning."""
        for group in self.param_groups:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None
            state = self.state[weight]

            g = weight.grad.data
            s = g.shape

            g = weight.grad.data
            weight.og_grad = g

            if group["layer_type"] == "Conv2d":
                g = g.contiguous().view(s[0], s[1] * s[2] * s[3])

            if bias is not None:
                gb = bias.grad.data
                bias.og_grad = gb
                g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)

            # compute precondition
            ngrad = g / (state["gg"] + self.eps)

            # precondition
            if bias is not None:
                gb = ngrad[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                ngrad = ngrad[:, :-1]

            ngrad = ngrad.contiguous().view(*s)
            weight.grad.data = ngrad

        self._iteration_counter += 1


def DKL_discrete(p1, p2):
    d_kl = (p1 * (torch.log(p1) - torch.log(p2))).sum(-1)
    return d_kl.mean()


def DKL_continuous(z0, z1):
    mu0, log_std0, std0 = z0
    mu1, log_std1, std1 = z1
    kl = (
        log_std1
        - log_std0
        + (std0.pow(2) + (mu0 - mu1).pow(2)) / (2.0 * std1.pow(2))
        - 0.5
    )
    return kl.sum(-1).mean()


# BFGS not used in the paper but can be used if you want
import scipy.optimize


def bfgs_update_value(value_net, vloss_fcn):
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.from_numpy(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        vloss = vloss_fcn()
        vloss.backward()

        return (
            vloss.cpu().data.double().numpy(),
            get_flat_grad_from(value_net).cpu().data.double().numpy(),
        )

    flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        vec(value_net.parameters()).cpu().detach().double().numpy(),
        maxiter=25,
    )
    set_flat_params_to(value_net, torch.from_numpy(flat_params))


def conjugate_gradients(Avp, b, nsteps, device, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device)
    r = b.clone().to(device)
    p = b.clone().to(device)
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(
    model,
    f,
    x,
    fullstep,
    expected_improve_rate,
    max_backtracks=10,
    accept_ratio=0.1,
    verbose=False,
):
    fval = f().data
    if verbose:
        print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f().data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if verbose:
            print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            if verbose:
                print("fval after", newfval.item())
            return True, xnew
    return False, x


class HessianFree(Optimizer):
    def __init__(self, net, damping, max_kl, n_cg_steps):
        self.params = list(net.parameters())
        self.max_kl = max_kl
        self.damping = damping
        self.n_cg_steps = n_cg_steps
        self._name = "hessianfree"
        super().__init__(self.params, {})

    def step(self, output_space_fcn, loss):
        # compute gradients
        grads = torch.autograd.grad(loss, self.params, retain_graph=True)
        fgrads = vec(grads)

        def vp(v):
            kl = output_space_fcn()
            grads = torch.autograd.grad(kl, self.params, create_graph=True)  # J
            fgrads = vec(grads)

            Jv = (fgrads * v).sum()  # Jv
            Hv = torch.autograd.grad(Jv, self.params)  # grad(Jv) = Hv
            flatHv = vec(Hv)

            return flatHv + v * self.damping

        stepdir = conjugate_gradients(vp, -fgrads, self.n_cg_steps, fgrads.device)

        shs = 0.5 * (stepdir * vp(stepdir)).sum()
        lm = torch.sqrt(shs / self.max_kl)
        alpha = 1 / lm

        neggdotstepdir = (-fgrads * stepdir).sum(0, keepdim=True)

        return alpha, stepdir, neggdotstepdir


class EKFAC(Optimizer):
    def __init__(self, net, eps, sua=False, ra=False, update_freq=1, alpha=0.75):
        """EKFAC Preconditionner for Linear and Conv2d layers.
        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            ra (bool): Computes stats using a running average of averaged gradients
                instead of using a intra minibatch estimate
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter
        """
        self._name = "ekfac"
        self.eps = eps
        self.sua = sua
        self.ra = ra
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.acc_stats = False
        if not self.ra and self.alpha != 1.0:
            raise NotImplementedError

        # setup hooks
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ["Linear", "Conv2d"]:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {"params": params, "mod": mod, "layer_type": mod_class}
                if mod_class == "Conv2d":
                    if not self.sua:
                        # Adding gathering filter for convolution
                        d["gathering_filter"] = self._get_gathering_filter(mod)
                self.params.append(d)
        super(EKFAC, self).__init__(self.params, {})

    def start_acc_stats(self):
        self.acc_stats = True

    def stop_acc_stats(self):
        self.acc_stats = False

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]["a"] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if self.acc_stats:
            self.state[mod]["g"] = grad_output[0] * grad_output[0].size(0)

    def step(self):
        """Performs one step of preconditioning."""
        for group in self.param_groups:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None
            state = self.state[weight]

            # save og grads
            weight.og_grad = weight.grad.data
            if bias is not None:
                bias.og_grad = bias.grad.data

            # Update convariances and inverses
            if self._iteration_counter % self.update_freq == 0:
                self._compute_kfe(group, state)

            # Preconditionning
            if group["layer_type"] == "Conv2d" and self.sua:
                if self.ra:
                    self._precond_sua_ra(weight, bias, group, state)
                else:
                    self._precond_intra_sua(weight, bias, group, state)
            else:
                if self.ra:
                    self._precond_ra(weight, bias, group, state)  # !!
                else:
                    self._precond_intra(weight, bias, group, state)
        self._iteration_counter += 1

    def _precond_ra(self, weight, bias, group, state):
        """Applies preconditioning."""
        Q_a = state["Q_a"]
        Q_g = state["Q_g"]
        m2 = state["m2"]
        g = weight.grad.data
        s = g.shape
        bs = self.state[group["mod"]]["a"].size(0)
        if group["layer_type"] == "Conv2d":
            g = g.contiguous().view(s[0], s[1] * s[2] * s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        # project
        g_kfe = Q_g.t() @ g @ Q_a
        # scale (m2 = d_g.unsqueeze(1) * d_a.unsqueeze(0))
        # m2 * alpha + [(1 - alpha) * batchsize] * g_kfe**2
        m2.mul_(self.alpha).add_((1.0 - self.alpha) * bs, g_kfe ** 2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        # project back
        g_nat = Q_g @ g_nat_kfe @ Q_a.t()

        # precondition
        if bias is not None:
            gb = g_nat[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        g_nat = g_nat.contiguous().view(*s)
        weight.grad.data = g_nat

    def _compute_kfe(self, group, state):
        """Computes the covariances."""
        mod = group["mod"]
        a = self.state[group["mod"]]["a"]
        g = self.state[group["mod"]]["g"]

        # Computation of A
        if group["layer_type"] == "Conv2d":
            if not self.sua:
                a = F.conv2d(
                    a,
                    group["gathering_filter"].to(a.device),
                    stride=mod.stride,
                    padding=mod.padding,
                    groups=mod.in_channels,
                )
            a = a.data.permute(1, 0, 2, 3).contiguous().view(a.shape[1], -1)
        else:
            a = a.data.t()

        if mod.bias is not None:
            ones = torch.ones_like(a[:1])
            a = torch.cat([a, ones], dim=0)

        A = a @ (a.t() / float(a.shape[1]))
        d_a, state["Q_a"] = torch.symeig(A, eigenvectors=True)
        # d_a, state['Q_a'] = torch.linalg.eigh(A)

        # Computation of gg
        if group["layer_type"] == "Conv2d":
            g = g.data.permute(1, 0, 2, 3)
            state["num_locations"] = g.shape[2] * g.shape[3]
            g = g.contiguous().view(g.shape[0], -1)
        else:
            g = g.data.t()
            state["num_locations"] = 1

        gg = g @ (g.t() / float(g.shape[1]))
        d_g, state["Q_g"] = torch.symeig(gg, eigenvectors=True)
        # d_g, state['Q_g'] = torch.linalg.eigh(gg)

        state["m2"] = d_g.unsqueeze(1) * d_a.unsqueeze(0) * state["num_locations"]
        if group["layer_type"] == "Conv2d" and self.sua:
            ws = group["params"][0].grad.data.size()
            state["m2"] = (
                state["m2"]
                .view(d_g.size(0), d_a.size(0), 1, 1)
                .expand(-1, -1, ws[2], ws[3])
            )

    def _precond_intra(self, weight, bias, group, state):
        """Applies preconditioning."""
        Q_a = state["Q_a"]
        Q_g = state["Q_g"]
        mod = group["mod"]
        x = self.state[mod]["a"]
        gy = self.state[mod]["g"]
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_cin = 0
        s_gy = gy.size()
        bs = x.size(0)
        if group["layer_type"] == "Conv2d":
            x = F.conv2d(
                x,
                group["gathering_filter"],
                stride=mod.stride,
                padding=mod.padding,
                groups=mod.in_channels,
            )
            s_x = x.size()
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
                s_cin = 1  # adding a channel in dim for the bias
            # intra minibatch m2
            x_kfe = (
                torch.mm(Q_a.t(), x)
                .view(s_x[1] + s_cin, -1, s_x[2], s_x[3])
                .permute(1, 0, 2, 3)
            )
            gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
            gy_kfe = (
                torch.mm(Q_g.t(), gy)
                .view(s_gy[1], -1, s_gy[2], s_gy[3])
                .permute(1, 0, 2, 3)
            )
            m2 = torch.zeros((s[0], s[1] * s[2] * s[3] + s_cin), device=g.device)
            g_kfe = torch.zeros((s[0], s[1] * s[2] * s[3] + s_cin), device=g.device)
            for i in range(x_kfe.size(0)):
                g_this = torch.mm(
                    gy_kfe[i].view(s_gy[1], -1),
                    x_kfe[i].permute(1, 2, 0).view(-1, s_x[1] + s_cin),
                )
                m2 += g_this ** 2
            m2 /= bs
            g_kfe = (
                torch.mm(
                    gy_kfe.permute(1, 0, 2, 3).view(s_gy[1], -1),
                    x_kfe.permute(0, 2, 3, 1).contiguous().view(-1, s_x[1] + s_cin),
                )
                / bs
            )
            ## sanity check did we obtain the same grad ?
            # g = torch.mm(torch.mm(Q_g, g_kfe), Q_a.t())
            # gb = g[:,-1]
            # gw = g[:,:-1].view(*s)
            # print('bias', torch.dist(gb, bias.grad.data))
            # print('weight', torch.dist(gw, weight.grad.data))
            ## end sanity check
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(Q_g, g_nat_kfe), Q_a.t())
            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat
        else:
            if bias is not None:
                ones = torch.ones_like(x[:, :1])
                x = torch.cat([x, ones], dim=1)
            x_kfe = x @ Q_a
            gy_kfe = gy @ Q_g
            m2 = (gy_kfe.t() ** 2 @ x_kfe ** 2) / bs
            # (gy @ Q_g)T @ x @ Q_a = Q_g @ gy @ x @ Q_a = Q_g @ Dw @ Q_a
            g_kfe = (gy_kfe.t() @ x_kfe) / bs

            # m2 = (gy_kfe.t() @ x_kfe)**2 / bs  -- diff but still works
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = Q_g @ g_nat_kfe @ Q_a.t()

            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat

    def _precond_sua_ra(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        Q_a = state["Q_a"]
        Q_g = state["Q_g"]
        m2 = state["m2"]
        g = weight.grad.data
        s = g.shape
        bs = self.state[group["mod"]]["a"].size(0)
        mod = group["mod"]
        if bias is not None:
            gb = bias.grad.view(-1, 1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=1)
        g_kfe = self._to_kfe_sua(g, Q_a, Q_g)
        m2.mul_(self.alpha).add_((1.0 - self.alpha) * bs, g_kfe ** 2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, Q_a.t(), Q_g.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2] // 2, s[3] // 2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _precond_intra_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        Q_a = state["Q_a"]
        Q_g = state["Q_g"]
        mod = group["mod"]
        x = self.state[mod]["a"]
        gy = self.state[mod]["g"]
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_gy = gy.size()
        s_cin = 0
        bs = x.size(0)
        if bias is not None:
            ones = torch.ones_like(x[:, :1])
            x = torch.cat([x, ones], dim=1)
            s_cin += 1
        # intra minibatch m2
        x = x.permute(1, 0, 2, 3).contiguous().view(s_x[1] + s_cin, -1)
        x_kfe = (
            torch.mm(Q_a.t(), x)
            .view(s_x[1] + s_cin, -1, s_x[2], s_x[3])
            .permute(1, 0, 2, 3)
        )
        gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
        gy_kfe = (
            torch.mm(Q_g.t(), gy)
            .view(s_gy[1], -1, s_gy[2], s_gy[3])
            .permute(1, 0, 2, 3)
        )
        m2 = torch.zeros((s[0], s[1] + s_cin, s[2], s[3]), device=g.device)
        g_kfe = torch.zeros((s[0], s[1] + s_cin, s[2], s[3]), device=g.device)
        for i in range(x_kfe.size(0)):
            g_this = grad_wrt_kernel(
                x_kfe[i : i + 1], gy_kfe[i : i + 1], mod.padding, mod.stride
            )
            m2 += g_this ** 2
        m2 /= bs
        g_kfe = grad_wrt_kernel(x_kfe, gy_kfe, mod.padding, mod.stride) / bs
        ## sanity check did we obtain the same grad ?
        # g = self._to_kfe_sua(g_kfe, Q_a.t(), Q_g.t())
        # gb = g[:, -1, s[2]//2, s[3]//2]
        # gw = g[:,:-1].view(*s)
        # print('bias', torch.dist(gb, bias.grad.data))
        # print('weight', torch.dist(gw, weight.grad.data))
        ## end sanity check
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, Q_a.t(), Q_g.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2] // 2, s[3] // 2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh * j + kw * kh * i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg):
        """Project g to the kfe"""
        sg = g.size()
        g = torch.mm(vg.t(), g.view(sg[0], -1)).view(vg.size(1), sg[1], sg[2], sg[3])
        g = torch.mm(g.permute(0, 2, 3, 1).contiguous().view(-1, sg[1]), vx)
        g = g.view(vg.size(1), sg[2], sg[3], vx.size(1)).permute(0, 3, 1, 2)
        return g

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()


def grad_wrt_kernel(a, g, padding, stride, target_size=None):
    gk = F.conv2d(
        a.transpose(0, 1),
        g.transpose(0, 1).contiguous(),
        padding=padding,
        dilation=stride,
    ).transpose(0, 1)
    if target_size is not None and target_size != gk.size():
        return gk[:, :, : target_size[2], : target_size[3]].contiguous()
    return gk


class KFAC(Optimizer):
    def __init__(
        self,
        net,
        eps,
        sua=False,
        pi=False,
        update_freq=1,
        alpha=1.0,
        constraint_norm=False,
    ):
        """K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self._name = "kfac"
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.acc_stats = False
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ["Linear", "Conv2d"]:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {"params": params, "mod": mod, "layer_type": mod_class}
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def start_acc_stats(self):
        self.acc_stats = True

    def stop_acc_stats(self):
        self.acc_stats = False

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.0
        for group in self.param_groups:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None
            state = self.state[weight]

            # save og grads
            weight.og_grad = weight.grad.data
            if bias is not None:
                bias.og_grad = bias.grad.data

            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(
                        state["xxt"], state["ggt"], state["num_locations"]
                    )
                    state["ixxt"] = ixxt
                    state["iggt"] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)
            if update_params:
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
            # Cleaning
            if "x" in self.state[group["mod"]]:
                del self.state[group["mod"]]["x"]
            if "gy" in self.state[group["mod"]]:
                del self.state[group["mod"]]["gy"]
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1.0 / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group["params"]:
                    param.grad.data *= scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]["x"] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if self.acc_stats:
            self.state[mod]["gy"] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group["layer_type"] == "Conv2d" and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        if group["layer_type"] == "Conv2d":
            g = g.contiguous().view(s[0], s[1] * s[2] * s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group["layer_type"] == "Conv2d":
            g /= state["num_locations"]
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state["ixxt"]
        iggt = state["iggt"]
        g = weight.grad.data
        s = g.shape
        mod = group["mod"]
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0] * s[2] * s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state["num_locations"]
        if bias is not None:
            gb = g[:, -1, s[2] // 2, s[3] // 2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group["mod"]
        x = self.state[group["mod"]]["x"]
        gy = self.state[group["mod"]]["gy"]
        # Computation of xxt
        if group["layer_type"] == "Conv2d":
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state["xxt"] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state["xxt"].addmm_(
                mat1=x,
                mat2=x.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(x.shape[1]),
            )
        # Computation of ggt
        if group["layer_type"] == "Conv2d":
            gy = gy.data.permute(1, 0, 2, 3)
            state["num_locations"] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state["num_locations"] = 1
        if self._iteration_counter == 0:
            state["ggt"] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state["ggt"].addmm_(
                mat1=gy,
                mat2=gy.t(),
                beta=(1.0 - self.alpha),
                alpha=self.alpha / float(gy.shape[1]),
            )

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = tx / tg
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
