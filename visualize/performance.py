#%%
import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import tsplot, path2valuestimes
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

paths_and_names = [
    ("diagonal_runs/", "Diagonal"),
    ("hessianfree_runs/", "HF"),
    ("kfac_runs/", "KFAC"),
    ("ekfac_runs/", "EKFAC"),
    ("tengradv2_runs/", "TENGraD"),
]

# %%
def analyze(runs_path, speed_runs_path):
    bp, suffix = runs_path, ""
    f = lambda hp: True

    #### make it look nice
    env_paths = [
        pathlib.Path(f"{bp}/HalfCheetah-v2{suffix}/"),
        pathlib.Path(f"{bp}/Ant-v2{suffix}/"),
        pathlib.Path(f"{bp}/Hopper-v2{suffix}/"),
        pathlib.Path(f"{bp}/Humanoid-v2{suffix}/"),
        pathlib.Path(f"{bp}/Walker2d-v2{suffix}/"),
        pathlib.Path(f"{bp}/Reacher-v2{suffix}/"),
        pathlib.Path(f"{bp}/Swimmer-v2{suffix}/"),
    ]

    limit = {
        "HalfCheetah-v2": [-1000],
        "Ant-v2": [-1000, 3000],
        "Hopper-v2": [-100],
        "Humanoid-v2": [0],
        "Walker2d-v2": [-100],
        "Reacher-v2": [-150, 20],
        "Swimmer-v2": [-20],
    }

    plot_time = False
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    fvalues_performance = {}
    fvalues_stability = {}
    fvalues_threshold = {}
    fvalues_seeds = {}

    if not os.path.isfile("json/performance_thresholds.json"):
        assert (
            name == "baseline"
        ), "thresholds should be extracted from baseline performance runs"

        ## extract threshold values
        env_2_threshold = {}
        for i, (ax, rp) in enumerate(zip(axes.flatten(), env_paths)):
            print(f"----- {rp} -----")
            data_and_names = []
            for main_path, n in paths_and_names:
                data = (path2valuestimes(rp, main_path, f, hparams=True), n)
                data_and_names.append(data)

            # the best mean score of the lowest ranking optimizer
            threshold = min([v.mean(0).max(0) for (v, _, _, _), _ in data_and_names])
            env_2_threshold[rp.name] = threshold

        # save the thresholds for future comparison
        print("saving performance threshold values...")
        with open("json/performance_thresholds.json", "w") as file:
            json.dump(env_2_threshold, file, indent=4, sort_keys=True)
    else:
        with open("json/performance_thresholds.json", "r") as file:
            env_2_threshold = json.load(file)

    for i, (ax, rp) in enumerate(zip(axes.flatten(), env_paths)):
        print(f"----- {rp} -----")
        data_and_names = []
        for main_path, n in paths_and_names:
            data = (path2valuestimes(rp, main_path, f, hparams=True), n)
            data_and_names.append(data)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(data_and_names)))

        colidx = str(rp.name).split("_")[0]
        fvalues_threshold[colidx] = {}
        fvalues_performance[colidx] = {}
        fvalues_stability[colidx] = {}
        fvalues_seeds[colidx] = {}

        threshold = env_2_threshold[colidx]

        for ((values, _, steps, hps), n), c in zip(data_and_names, colors):

            time = steps[0]
            time *= hps[0]["num_envs"] * hps[0]["num_steps"]

            std = values.std(0)
            # ci_95 = 1.96 * std / (values.shape[0] ** 0.5)
            fvalues_stability[colidx][n] = -std.mean(0)  # larger values = worse
            fvalues_performance[colidx][n] = values.mean(0).max(
                0
            )  # larger performance = better
            fvalues_seeds[colidx][n] = len(hps)

            idx = (values.mean(0) >= threshold).nonzero()[0]
            if len(idx) > 0:
                idx = idx[0]
                fvalues_threshold[colidx][n] = (
                    -time[idx] / 1000.0
                )  # more env steps = worse score
            else:
                fvalues_threshold[colidx][n] = np.nan

            tsplot(values, time, ax=ax, label=f"{n}", color=c)

        env_name = rp.name
        ax.set_title(env_name)

        if len(limit[env_name]) == 1:
            ax.set_ylim(bottom=limit[env_name][0])
        else:
            ax.set_ylim(limit[env_name])

        # ax.set_xlim([0, 1e6])

        if i == 0:
            ax.set_xlabel("Time (ms)" if plot_time else "Number of Environment Steps")
            ax.set_ylabel("Average Return")
            ax.legend()

    axes.flatten()[-1].axis('off')
    for ax in axes.flatten()[-4:-1]:
        pos = ax.get_position()
        pos.x0 += 0.1       
        pos.x1 += 0.1       
        ax.set_position(pos)
    
    fig.savefig("performance.pdf", bbox_inches='tight')
    plt.show()

    speed = get_speed_df(speed_runs_path)

    #%%
    perf = pd.DataFrame(data=fvalues_performance).T
    stab = pd.DataFrame(data=fvalues_stability).T
    threshold = pd.DataFrame(data=fvalues_threshold).T
    seeds = pd.DataFrame(data=fvalues_seeds).T

    return perf, stab, threshold, speed, seeds


# normalize performance of each env so the scores are equally weighted
def get_normalized_perf(perf):
    for (c, maxv), (c2, minv) in zip(
        perf.max(axis=1).items(), perf.min(axis=1).items()
    ):
        assert c == c2
        perf.loc[c] = (perf.loc[c] - minv) / (maxv - minv)
    scores = (perf.mean(0) * 100).round(2)
    return scores


def mean_df(perf, stab, threshold, speed, seeds, name, save=False):
    pathlib.Path(f"csv/{name}").mkdir(exist_ok=True, parents=True)

    names = ["perf", "stab", "threshold", "speed"]
    if save:
        for metric_n, df in zip(names, [perf, stab, threshold, speed]):
            df.to_csv(f"csv/{name}/{name}_{metric_n}.csv")

    data = {
        "Performance": dict(perf.mean(0)),
        "Stability": dict(stab.mean(0)),
        "Sample Efficiency": dict(threshold.mean(0)),
        "Speed": dict(speed.mean(0)),
        "Norm_Performance": dict(get_normalized_perf(perf.copy()))
        # 'Seeds': dict(seeds.mean(0)),
    }
    df = pd.DataFrame(data)

    # df = df.round(0)
    for c in df.columns:
        if c == "Speed":
            df[c] = df[c].round(3)
        elif c == "Norm_Performance":
            df[c] = df[c].round(2)
        elif not (c == "Seeds"):
            df[c] = df[c].round(0)

    if save:
        df.to_csv(f"csv/{name}/{name}_all.csv")

    return df


def load(name):
    perf = pd.read_csv(f"csv/{name}/{name}_perf.csv", index_col=0)
    stab = pd.read_csv(f"csv/{name}/{name}_stab.csv", index_col=0)
    threshold = pd.read_csv(f"csv/{name}/{name}_threshold.csv", index_col=0)
    speed = pd.read_csv(f"csv/{name}/{name}_speed.csv", index_col=0)
    df = pd.read_csv(f"csv/{name}/{name}_all.csv", index_col=0)
    return (perf, stab, threshold, speed), df


def get_speed_df(runs_path):
    bp, suffix = runs_path, ""
    f = lambda hp: True
    env_paths = [
        pathlib.Path(f"{bp}/HalfCheetah-v2{suffix}/"),
        pathlib.Path(f"{bp}/Ant-v2{suffix}/"),
        pathlib.Path(f"{bp}/Hopper-v2{suffix}/"),
        pathlib.Path(f"{bp}/Humanoid-v2{suffix}/"),
        pathlib.Path(f"{bp}/Walker2d-v2{suffix}/"),
        pathlib.Path(f"{bp}/Reacher-v2{suffix}/"),
        pathlib.Path(f"{bp}/Swimmer-v2{suffix}/"),
    ]
    fvalues_time = {}
    for i, rp in enumerate(env_paths):
        colidx = str(rp.name).split("_")[0]
        fvalues_time[colidx] = {}

        data_and_names = []
        for main_path, n in paths_and_names:
            data = (path2valuestimes(rp, main_path, f, hparams=True), n)
            data_and_names.append(data)

        for (values, times, steps, hps), n in data_and_names:
            time = times[0][-1]
            fvalues_time[colidx][n] = -time
    speed = pd.DataFrame(data=fvalues_time).T
    return speed


# %%
read_csv_data = False
save = False

# %%
runs_path = pathlib.Path("../runs/5e6_baseline/")
speed_runs_path = pathlib.Path("../runs/baseline_speed/")
name = "baseline"
if read_csv_data:
    baseline_data, baseline_df = load(name)
else:
    baseline_data = analyze(runs_path, speed_runs_path)
    baseline_df = mean_df(*baseline_data, name, save=save)
baseline_df
# latex(baseline_df)

#%%
runs_path = pathlib.Path("../runs/5e6_best_batch/")
speed_runs_path = pathlib.Path("../runs/batch_speed/")
name = "best_batch"
if read_csv_data:
    batch_data, batch_df = load(name)
else:
    batch_data = analyze(runs_path, speed_runs_path)
    batch_df = mean_df(*batch_data, name, save=save)
batch_df

#%%
runs_path = pathlib.Path("../runs/5e6_best_critic/")
speed_runs_path = pathlib.Path("../runs/critic_speed/")
name = "best_critic"
if read_csv_data:
    critic_data, critic_df = load(name)
else:
    critic_data = analyze(runs_path, speed_runs_path)
    critic_df = mean_df(*critic_data, name, save=save)
critic_df

# %%
def full_summary(data1, data2):
    names = ["perf", "stab", "threshold"]
    dfs = []
    for n in names:
        df = percent_improvement_metric(n, data1, data2)
        dfs.append(df)

    df = df.copy()
    for (i, p_row), (j, s_row), (k, t_row) in zip(*[d.iterrows() for d in dfs]):
        for p, s, t in zip(p_row.items(), s_row.items(), t_row.items()):
            row, p = p
            s, t = s[1], t[1]
            p, s, t = [v.split("(")[-1][:-1] for v in (p, s, t)]
            df[row][i] = f"({p}, {s}, {t})"
    return df


# improvement from df1 -> df2
def percent_improvement(df1, df2):
    df_improve = (df2.abs() - df1.abs()) / df1 * 100.0
    for (i, r_base_df2), (j, r_improv), (l, r_base_df1) in zip(
        df2.iterrows(), df_improve.iterrows(), df1.iterrows()
    ):
        for k in r_base_df2.keys():
            if np.isnan(r_base_df2[k]):  # df2 doesnt acheive score / baseline
                r_base_df2[k] = f"NaN"
            elif np.isnan(r_base_df1[k]):  # df1 achieves score but df2 doesnt
                r_base_df2[k] = f"{r_base_df2[k] :.0f} (!)"
            else:
                r_base_df2[
                    k
                ] = f"{r_base_df2[k] :.0f} ({'+' if r_improv[k] >= 0 else ''}{r_improv[k] :.0f}%)"
        df_improve.loc[j] = r_base_df2
    return df_improve


col2idx = {
    "perf": 0,
    "stab": 1,
    "threshold": 2,
    "speed": 3,
}


def percent_improvement_metric(data1, data2, col):
    idx = col2idx[col]
    return percent_improvement(data1[idx], data2[idx])


#%%
def full_improvements_approx(approx_name, data1, data2):
    col_names = ["Performance", "Stability", "Sample Efficiency", "Speed"]
    metric_names = ["perf", "stab", "threshold", "speed"]
    dfs = []
    for col_name, metric_name in zip(col_names, metric_names):
        df = percent_improvement_metric(data1, data2, metric_name)
        df = df[[approx_name]]
        df.columns = [col_name]
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df


def latex(df):
    print(df.to_latex())


def rliable_aggregate_metrics():
    runs_path = pathlib.Path("../runs/5e6_baseline/")
    bp, suffix = runs_path, ""
    f = lambda _: True
    env_paths = [
        pathlib.Path(f"{bp}/HalfCheetah-v2{suffix}/"),
        pathlib.Path(f"{bp}/Ant-v2{suffix}/"),
        pathlib.Path(f"{bp}/Hopper-v2{suffix}/"),
        pathlib.Path(f"{bp}/Humanoid-v2{suffix}/"),
        pathlib.Path(f"{bp}/Walker2d-v2{suffix}/"),
        pathlib.Path(f"{bp}/Reacher-v2{suffix}/"),
        pathlib.Path(f"{bp}/Swimmer-v2{suffix}/"),
    ]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths_and_names)))

    # algorithm -> env scores dict
    color_dict = {}
    data_and_names = {}
    for (main_path, n), col in zip(paths_and_names, colors):
        data_and_names[n] = []
        color_dict[n] = col
        for i, rp in enumerate(env_paths):
            (values, _, steps, hps) = path2valuestimes(rp, main_path, f, hparams=True)
            data_and_names[n].append(values.max(-1))

    # normalize the scores based on best total score in env
    for i in range(len(env_paths)):
        best_score = -float("inf")
        worst_score = float("inf")
        for n in data_and_names:
            best_score = max(data_and_names[n][i].max(), best_score)
            worst_score = min(data_and_names[n][i].min(), worst_score)

        for n in data_and_names:
            data_and_names[n][i] = (data_and_names[n][i] - worst_score) / (
                best_score - worst_score
            )

    # n_runs x n_games matrix
    for n in data_and_names:
        data_and_names[n] = np.array(data_and_names[n]).T

    #%%
    # compute the results
    algorithms = list(data_and_names.keys())

    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        data_and_names, aggregate_func, reps=50000
    )

    #%%
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=algorithms,
        xlabel="Normalized Performance Scores",
        colors=color_dict,
        xlabel_y_coordinate=-0.18,
    )

    fig.savefig("metrics.pdf", bbox_inches='tight')
    return fig 

#%%
fig = rliable_aggregate_metrics()
plt.show()

#%%
if save:
    save_path = pathlib.Path("csv/improvements/")
    save_path.mkdir(exist_ok=True, parents=True)
    (save_path / "batch/").mkdir(exist_ok=True)
    (save_path / "critic/").mkdir(exist_ok=True)

    for _, approx_name in paths_and_names:
        df = full_improvements_approx(approx_name, baseline_data, batch_data)
        df.to_csv(f"{save_path}/batch/batch_improve_{approx_name}.csv")

        df = full_improvements_approx(approx_name, baseline_data, critic_data)
        df.to_csv(f"{save_path}/critic/critic_improve_{approx_name}.csv")

#%%
print("--- BATCH SIZE IMPROVEMENTS ---")
for _, approx_name in paths_and_names:
    print(f"---- {approx_name} ----")
    df = full_improvements_approx(approx_name, baseline_data, batch_data)
    latex(df)

#%%
print("--- CRITIC + BATCH SIZE IMPROVEMENTS ---")
for _, approx_name in paths_and_names:
    print(f"---- {approx_name} ----")
    df = full_improvements_approx(approx_name, baseline_data, critic_data)
    latex(df)

# %%
print(percent_improvement(baseline_df, critic_df).to_latex())

# %%
percent_improvement(batch_df, critic_df)

# %%
percent_improvement_metric("perf", baseline_data, batch_data)

#%%
percent_improvement_metric("perf", batch_data, critic_data)
