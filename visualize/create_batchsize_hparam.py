#%%
import json

path = "hparams.json"  ## performance json

with open(path, "r") as f:
    hparams = json.load(f)

# %%
for approx in hparams:
    hp = hparams[approx]
    hp.pop("num_envs")  # remove num_envs (we'll hparam search over this)
    hparams[approx] = hp

# %%
with open("batch_hparams.json", "w") as f:
    json.dump(hparams, f, indent=4, sort_keys=True)

# %%
