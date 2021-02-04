
# Generative Adversarial Reinforcement learning


This is code for GAenerative Adversarial Reinforment Leanring Framework on the environments Coinrun along with example scripts.

Authors: Qi Yang, Peng Yang, Ke Tang

![CoinRun](coinrun.png?raw=true "CoinRun")

## Install

You should install the package in development mode so you can easily change the files.  You may also want to create a virtualenv before installing since it depends on a specific version of OpenAI baselines.

This environment has been used on Mac OS X and Ubuntu 16.04 with Python 3.6.

```
# Linux
apt-get install mpich build-essential qt5-default pkg-config
# Mac
brew install qt open-mpi pkg-config

git clone https://github.com/Desein-Yang/GARL.git
cd coinrun
pip install tensorflow==1.12.0  # or tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```

Note that this does not compile the environment, the environment will be compiled when the `coinrun` package is imported.

## Usage

Train an agent using GARL + PPO:

```
python -m garl.train_agent --run-id myrun --save-interval 1
```

After each parameter update, this will save a copy of the agent to `./logs/{runid}.log`. Results are logged to `/tmp/tensorflow` by default.

Run parallel training using MPI:

```
mpiexec -np 8 python -m garl.train_agent --run-id myrun
```

Evaluate an agent's final training performance across N parallel environments. Evaluate K levels on each environment (NxK total levels). Default N=20 is reasonable. Evaluation levels will be drawn from the same set as those seen during training.

```
python enjoy_2.py --train-eval --restore-id myrun -num-eval N -rep K
```

Evaluate an agent's final test performance on PxNxK distinct levels. All evaluation levels are uniformly sampled from the set of all high difficulty levels. Although we don't explicitly enforce that the test set avoid training levels, the probability of collisions is negligible.

View training options:

```
python -m coinrun.train_agent --help
```

Train an agent to play CoinRun-Platforms, using a larger number of environments to stabilize learning:

```
python train_agent.py --run-id coinrun_plat --game-type platform --num-envs 96 --use-lstm 1
```
`train_tasks` contains `SeedOptimizaer` to optimize train-set and `ppo_v4.py` contains `learn` to optimize policy with PPO.
You can change fitness or mutate operators on `train_task.py`.
## Docker

There's also a `Dockerfile` to create a CoinRun docker image:

```
docker build --tag coinrun .
docker run --rm coinrun python -um coinrun.random_agent
```

## Ref

1. [Quantifying Generalization in Reinforcement Learning](https://drive.google.com/file/d/1U1-uufB_ZzQ1HG67BhW9bB8mTJ6JtS19/view).

