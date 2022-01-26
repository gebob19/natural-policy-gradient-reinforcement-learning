#%%
import json
import pathlib
import pandas as pd
from utils import path2valuestimes

smooth = 0  # value smoothing
runs_path = pathlib.Path("runs/hparams_baseline/")

paths_and_names = [
    ("diagonal_runs/", "Diagonal"),
    ("hessianfree_runs/", "HF"),
    ("kfac_runs/", "KFAC"),
    ("ekfac_runs/", "EKFAC"),
    ("tengradv2_runs/", "TENGraD"),
]

# %%
#### HPARAM RESULTS
data_and_names = []
for main_path, n in paths_and_names:
    data = (path2valuestimes(runs_path, main_path, hparams=True), n)
    data_and_names.append(data)
print(len(data_and_names))

# %%
dataframes = []
columns = [
    "linesearch",
    "damping",
    "value_lr",
    "lr_max",
    "num_envs",
    "optim_name",
    "reward",
]

for (values, t, s, hps), n in data_and_names:
    pddata = []
    for i in range(len(hps)):
        hp = hps[i]
        v = values[i]
        row = (
            [float(hp[c]) for c in columns[:-2]] + [hp["optim_name"]] + [v[-10:].mean()]
        )
        pddata.append(row)

    df = pd.DataFrame(data=pddata, columns=columns)
    df = df.sort_values("reward", ascending=False)
    dataframes.append((df, n))

    print(f"best results: {n}")
    print(df.head())

# %%
best_params = {}
for df, n in dataframes:
    best_params[df.iloc[0].optim_name] = dict(df.iloc[0])

with open("visualize/json/baseline.json", "w") as f:
    f.write(json.dumps(best_params, indent=4, sort_keys=True))

# %%
