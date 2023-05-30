# Federated Learning under Covariate Shifts with Generalization Guarantees

This is the official code for [Federated Learning under Covariate Shifts with Generalization Guarantees](https://openreview.net/forum?id=N7lCDaeNiS) accepted at TMLR 2023.

## Installation 

```
conda create -n dafl python=3.7
conda activate dafl
pip install -r requirements.txt
python setup.py develop
wandb login
```


## Usage

The library has the following executables:

- `dafl/runner_target_shift.py`
- `dafl/runner_covariate_shift.py`
- `dafl/runner_ratio_estimation.py`
- `dafl/runner_upper_bound_estimate.py`


Example scripts:

```
python dafl/runner_upper_bound_estimate.py --num-clusters 10
python dafl/runner_target_shift.py --dataset fmnist --shift 5clients --batch-size 64 --num-steps 5000 --client-mode multi --wandb-name "TS/fmnist/shift=5clients/FedAvg"
python dafl/runner_target_shift.py --dataset fmnist --shift 5clients --batch-size 64 --num-steps 5000 --client-mode multi --use-true-ratio --wandb-name "TS/fmnist/shift=5clients/FITW-ERM"
python dafl/runner_target_shift.py --dataset fmnist --shift 5clients --batch-size 64 --num-steps 5000 --client-mode multi --use-true-ratio --combine-testsets --wandb-name "TS/fmnist/shift=5clients/FTW-ERM"
```

## Debugging

Useful prefixes:

```
OMP_NUM_THREADS=1 python dafl/runner.py
WANDB_MODE=dryrun python ...
```

- Use `OMP_NUM_THREADS` to avoid warning on local machine while testing
- Use `WANDB_MODE=dryrun` to not log to wandb

