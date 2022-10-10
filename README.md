# Understanding the Effects of Second-Order Approximations in Natural Policy Gradient Reinforcement Learning [[ArXiv]](https://arxiv.org/abs/2201.09104)

## Setup 

- Python 3.8.0
- `pip install -r req.txt`
- Mujoco 200 license

## Main Files

- `main.py`: main run file for model training 
- `models.py`: neural networks for policy and critic models 
- `optim.py`: second-order approximations for realizing the natural gradient 
- `utils.py`: helper functions 

## Reproducing Experiments

- `scripts/`: bash training scripts formatted for compute canada/SLURM jobs 
- `visualize/json`: training hyperparameters for each experiment 
- `visualize/csv`: training results in .csv format 
- `visualize/performance.py`: (after training) view results & create .csv results
    - best to run with VSCode ipython cells 

### Experiment Example

To run the baseline experiments:
- Tune hparams: `bash scripts/hparams/baseline.sh`
    - runs will be saved in `runs/hparams_baseline/...`
- Extract best hparams from runs: `python baseline_hparams.py`
    - the best hparams will be saved in `visualize/json/baseline.json`
- Run training with hparams: `bash scripts/baseline/diagonal.sh`
    - runs will be saved in `runs/5e6_baseline/...`
- Run speed tests: `bash scripts/speed/baseline.sh`
    - runs will be saved in `runs/baseline_speed/...`
- View results: run interactive ipython in `visualize/performance.py`

```
# %%
runs_path = pathlib.Path("../runs/5e6_baseline/")
speed_runs_path = pathlib.Path("../runs/baseline_speed/")
name = "baseline"
baseline_data = analyze(runs_path, speed_runs_path)
baseline_df = mean_df(*baseline_data, name, save=True)
```

## Second-order Approximation References

### Implementations
- [HF](https://github.com/ikostrikov/pytorch-trpo) 
- [KFAC & EKFAC](https://github.com/Thrandis/EKFAC-pytorch)
- [TENGraD](https://arxiv.org/abs/2106.03947v2)

## Other

- Code formatted with [Black](https://black.readthedocs.io/en/stable/index.html)
- Experiment runs format: `runs/{experiment_name}/{env_name}/{approximation}_runs/{tensorboard folder}/...`
