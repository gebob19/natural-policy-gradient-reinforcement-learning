#%%
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import path2valuestimes

smooth = 0  # value smoothing
runs_path = pathlib.Path("../runs/hparams_batch_runs/")
runs_path = pathlib.Path("../runs/5e6_hparams_batch_runs/")

paths_and_names = [
    ("diagonal_runs/", "diagonal"),
    ("hessianfree_runs/", "hessianfree"),
    ("kfac_runs/", "kfac"),
    ("ekfac_runs/", "ekfac"),
    ("tengradv2_runs/", "tengradv2"),
]

# %%
col_name = "num_envs"
bs2data = {}
for main_path, n in paths_and_names:
    bs2data[n] = {}

for main_path, n in paths_and_names:
    if n == "tengradv2":
        bss = [10, 8, 6, 4]
    else:
        bss = [25, 20, 15, 10, 5]

    for bs in bss:
        f = lambda hp: hp[col_name] == bs
        data = (path2valuestimes(runs_path, main_path, f), n)
        bs2data[n][bs] = data

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 20))
data = {}
for ax, k in zip(axes.flatten(), bs2data.keys()):
    data[k] = {}
    for bs, (d, n) in bs2data[k].items():
        values, times, steps = d
        steps = steps[0] * 1000 * bs

        final_v = values[0, -1]  # optimize off final performance
        # std = values.std(0).mean(0) # optimize off stability

        data[k][bs] = final_v
        ax.plot(steps, values.mean(0), label=bs)
    ax.legend()
    ax.set_title(k)

plt.legend()
plt.show()
df = pd.DataFrame(data)
df

# %%
best = {}
for k in data.keys():
    bs_results = data[k]
    bs, value = zip(*bs_results.items())
    best_idx = np.array(value).argmax()
    best_num_envs = bs[best_idx]
    best[k] = best_num_envs
best

# %%
with open("hparams.json", "r") as f:
    d = json.load(f)

for k in d.keys():
    # update num of envs to the best
    best_num_envs = best[k]
    d[k]["num_envs"] = best_num_envs

with open("best_batch_hparams.json", "w") as f:
    json.dump(d, f, indent=4, sort_keys=True)

# %%
