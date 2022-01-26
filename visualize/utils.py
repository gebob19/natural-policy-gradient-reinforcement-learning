#%%
import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def tsplot(d, x=None, ax=None, label=None, color=None, ci=True):
    if ax is None:
        ax = plt
    line = d.mean(0)
    std = d.std(0)
    if x is None:
        x = np.arange(len(line))
    ax.plot(x, line, label=label, color=color)

    if ci:
        n = d.shape[0]
        ci_95 = 1.96 * std / (n ** 0.5)
        shade = ci_95
    else:
        shade = std

    ax.fill_between(x, line + shade, line - shade, alpha=0.2, color=color)


def get_info(runs_path, main_path, f=None):
    values, times, steps = [], [], []
    hparams = []
    for path in (runs_path / main_path).iterdir():

        try:
            dfile = open(path / "results.json", "r")
        except:
            continue

        try:
            with open(path / "time.json", "r") as file:
                optim_time = json.load(file)["mean_poptim_time"]
        except:
            optim_time = None

        data = json.load(dfile)
        dfile.close()
        v = np.array([d[0] for d in data])
        time = np.array([d[1] for d in data])
        time = time - time[0]  # get runtime
        step = np.array([d[2] for d in data])

        # if smooth > 1:
        #     y = np.ones(smooth)
        #     z = np.ones(len(v))
        #     v = np.convolve(v,y,'same') / np.convolve(z,y,'same')

        dfile = open(path / "hparams.json", "r")
        hp = json.load(dfile)
        dfile.close()

        hp["time"] = optim_time

        if f is not None and not f(hp):
            print("skipping", path)
            continue  # skip

        values.append(v)
        times.append(time)
        steps.append(step)
        hparams.append(hp)

    print(f"{main_path} found {len(values)} runs...")

    return values, times, steps, hparams


def path2valuestimes(runs_path, main_path, f=None, hparams=False, debug=False):
    values, times, steps, hp = get_info(runs_path, main_path, f)
    if debug:
        print([v.shape for v in values], [h["seed"] for h in hp])
    values = np.stack(values)
    times = np.stack(times)
    steps = np.stack(steps)
    if hparams:
        return values, times, steps, hp
    else:
        return values, times, steps
