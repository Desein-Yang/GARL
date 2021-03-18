# =======================================================
# Generative adversarial reinforcement learning main code
#
# @author yangqi
# =======================================================

import time
from mpi4py import MPI
import tensorflow as tf
import wandb
import joblib
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import numpy as np
import garl.main_utils as utils
from garl import setup_utils, policies_back, wrappers, ppo2
from garl.config import Config
from garl.setup_utils import setup_and_load
from garl.eval import eval_test, eval_set
from garl.train_task import TaskOptimizer, SeedOptimizer
mpi_print = utils.mpi_print

def learn_func(**kwargs):
    if Config.USE_EVO == 1:
        f = ppo2.learn(**kwargs)
    elif Config.USE_EVO == 2:
        # origin ppo
        from garl import ppo2_v4
        f = ppo2_v4.learn(**kwargs)
    return f

def make_general_env(num_env,seed=None,rand_seed=None):
    from garl.coinrunenv import make
    env = make(Config.GAME_TYPE, num_env)
    if Config.FRAME_STACK > 1:
        env = VecFrameStack(env, Config.FRAME_STACK)

    if Config.EPSILON_GREEDY > 0:
        env = wrappers.EpsilonGreedyWrapper(env, Config.EPSILON_GREEDY)

    if Config.MU_OP == 1:
        env = wrappers.RandSeedWrapper(env,None,Config.INI_LEVELS)
    elif Config.MU_OP == 2:
        env = wrappers.ParamWrapper(env)

    env = wrappers.EpisodeRewardWrapper(env)
    return env

def mpi_print_res(res):
    for key in res.keys():
        utils.mpi_print(key,res[key])

def main():
    args = setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    seed = int(time.time()) % 10000
    utils.mpi_print(seed * 100 + rank)
    set_global_seeds(seed * 100 + rank)

    # For wandb package to visualize results curves
    config = Config.get_args_dict()
    config['global_seed'] = seed
    wandb.init(
        name = config["run_id"],
        project="coinrun",
        notes=" GARL generate seed",
        tags=["try"],
        config=config
    )

    utils.setup_mpi_gpus()
    utils.mpi_print('Set up gpu',args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    eval_limit = Config.EVAL_STEP*10**6
    phase_eval_limit = int(eval_limit // Config.TRAIN_ITER)
    total_timesteps = int(Config.TOTAL_STEP*10**6)
    phase_timesteps = int((total_timesteps-eval_limit) // Config.TRAIN_ITER)

    with tf.Session(config=config):
        sess = tf.get_default_session()

        # init env
        nenv = Config.NUM_ENVS
        env = make_general_env(nenv,rand_seed=seed)
        utils.mpi_print('Set up env')

        policy = policies_back.get_policy()
        utils.mpi_print('Set up policy')

        optimizer = SeedOptimizer(env=env,logdir=Config.LOGDIR,
                                  spare_size = Config.SPA_LEVELS,
                                  ini_size = Config.INI_LEVELS,
                                  eval_limit = phase_eval_limit,
                                  train_set_limit = Config.NUM_LEVELS,
                                  load_seed = Config.LOAD_SEED,
                                  rand_seed = seed,rep=1, log=True)

        step_elapsed = 0
        t = 0

        if args.restore_id is not None:
            datapoints = Config.get_load_data('default')['datapoints']
            step_elapsed = datapoints[-1][0]
            optimizer.load()
            seed = optimizer.hist[-1]
            env.set_seed(seed)
            t = 16
            print('loadrestore')
            Config.RESTORE_ID = Config.get_load_data('default')['args']['run_id']
            Config.RUN_ID = Config.get_load_data('default')['args']['run_id'].replace('-','_')

        while(step_elapsed < (Config.TOTAL_STEP-1)*10**6):
        # ============ GARL =================
            # optimize policy
            mean_rewards, datapoints = learn_func(sess=sess,policy=policy,env=env,
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
                                                  start_timesteps = step_elapsed,
                                                  total_timesteps = phase_timesteps,
                                                  index = t)

            # test catestrophic forgetting
            if 'Forget' in Config.RUN_ID:
                last_set = list(env.get_seed_set())
                if t > 0:
                    curr_set = list(env.get_seed_set())
                    last_scores, _ = eval_test(sess, nenv, last_set,
                                 train=True, idx=None,
                                 rep_count=len(last_set))
                    curr_scores, _ = eval_test(sess, nenv, curr_set,
                                 train=True, idx=None,
                                 rep_count=len(curr_set))
                    tmp = set(curr_set).difference(set(last_set))
                    mpi_print("Forgetting Exp")
                    mpi_print("Last setsize",len(last_set))
                    mpi_print("Last scores",np.mean(last_scores),
                              "Curr scores",np.mean(curr_scores))
                    mpi_print("Replace count",len(tmp))

            # optimize env
            step_elapsed = datapoints[-1][0]
            if t < Config.TRAIN_ITER:
                best_rew_mean = max(mean_rewards)
                env,step_elapsed = optimizer.run(sess,env,step_elapsed,best_rew_mean)
            t += 1

        save_final_test = True
        if save_final_test:
            final_test = {}
            final_test['step_elapsed'] = step_elapsed
            train_set = env.get_seed()
            final_test['train_set_size'] = len(train_set)
            eval_log = eval_test(sess, nenv, train_set,train=True,is_high=False,rep_count=1000,log=True)
            final_test['Train_set'] = eval_log

            eval_log = final_test(sess, nenv, None,train=False,is_high=True,rep_count=1000,log=True)
            final_test['Test_set'] = eval_log
            joblib.dump(final_test,setup_utils.file_to_path('final_test'))

    env.close()

if __name__ == '__main__':
    main()

