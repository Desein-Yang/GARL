"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""
import time
from mpi4py import MPI
import tensorflow as tf
import wandb
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies_back, wrappers, ppo2, ppo2_nr
from coinrun.config import Config

def learn_func(**kwargs):
    if Config.USE_LSTM == 2:
        f = ppo2_nr.learn(**kwargs)
    else:
        f = ppo2.learn(**kwargs)
    return f

def main():
    args = setup_utils.setup_and_load()
    setup_utils.load_for_setup_if_necessary()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('size',size)

    # For wandb package to visualize results curves
    config = Config.get_args_dict()
    wandb.init(
        project="coinrun",
        notes=" baseline train",
        tags=["baseline",Config.RUN_ID.split('-')[0]],
        config=config
    )

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()
    utils.mpi_print('Set up gpu')
    utils.mpi_print(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # nenvs is how many envs run parallel on a cpu
    # VenEnv class allows parallel rollout
    nenvs = Config.NUM_ENVS
    total_timesteps = int(256*10**6)

    env = utils.make_general_env(nenvs, seed=rank)
    utils.mpi_print('Set up env')

    with tf.Session(config=config):
        env = wrappers.add_final_wrappers(env)

        policy = policies_back.get_policy()
        #policy = policies.get_policy()
        utils.mpi_print('Set up policy')

        learn_func(policy=policy,
                    env=env,
                    log_interval=args.log_interval,
                    save_interval=args.save_interval,
                    nsteps=Config.NUM_STEPS,
                    nminibatches=Config.NUM_MINIBATCHES,
                    lam=Config.GAE_LAMBDA,
                    gamma=Config.GAMMA,
                    noptepochs=Config.PPO_EPOCHS,
                    ent_coef=Config.ENTROPY_COEFF,
                    vf_coef=Config.VF_COEFF,
                    max_grad_norm=Config.MAX_GRAD_NORM,
                    lr=lambda f : f * Config.LEARNING_RATE,
                    cliprange=lambda f : f * Config.CLIP_RANGE,
                    total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()

