#%%
#%%
import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import tsplot, get_info, path2valuestimes

smooth = 0  # value smoothing
runs_path = pathlib.Path("../runs/5e6_hparams_critic_runs/")

paths_and_names = [
    ("diagonal_runs/", "diagonal"),
    ("hessianfree_runs/", "hessianfree"),
    ("kfac_runs/", "kfac"),
    ("ekfac_runs/", "ekfac"),
    ("tengradv2_runs/", "tengradv2"),
]

#%%
f = lambda _: True
data_and_names = []
for main_path, n in paths_and_names:
    data = (path2valuestimes(runs_path, main_path, f, hparams=True), n)
    data_and_names.append(data)
print(len(data_and_names))

#%%
dataframes = []
columns = [
    "value_damping",
    "value_linesearch",
    "value_lr_max",
    "value_update",
    "optim_name",
    "reward",
]

for (values, t, s, hps), n in data_and_names:
    pddata = []
    for i in range(len(hps)):
        hp = hps[i]
        v = values[i]
        row = (
            [float(hp[c]) for c in columns[:-3]]
            + [hp["value_update"], hp["optim_name"]]
            + [v[-10:].mean()]
        )
        pddata.append(row)

    df = pd.DataFrame(data=pddata, columns=columns)
    df = df.sort_values("reward", ascending=False)
    dataframes.append((df, n))

    print(f"best results: {n}")
    print(df.head())

# %%
with open("best_post_batchruns.json", "r") as f:
    best_params = json.load(f)

for df, n in dataframes:
    optim_name = df.iloc[0].optim_name
    for k, v in df.iloc[0].items():
        if k == "value_linesearch":
            v = bool(v)
        best_params[optim_name][k] = v
best_params

# %%
with open("best_critic_runs.json", "w") as f:
    json.dump(best_params, f, indent=4, sort_keys=True)

# %%
# %%
