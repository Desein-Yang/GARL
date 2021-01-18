# =======================================================
# Generative adversarial reinforcement learning main code
#
# Modification:
# 1. add RandSeedWrapper() to control env seed
# 2. add progressive difficulty env
# 3. optimize parameters at T phase
# copyright QiYANG
# =======================================================
import time
from mpi4py import MPI
import tensorflow as tf
import wandb
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import garl.main_utils as utils
from garl import setup_utils, policies_back, wrappers, ppo2
from garl.config import Config
from garl.coinrunenv import make,setup_and_load
from garl.eval import eval_set

def learn_func(**kwargs):
    if Config.USE_EVO == 1:
        f = ppo2.learn(**kwargs)
    else:
        # origin ppo
        f = ppo2.learn(**kwargs)
    return f

def make_general_env(num_env,seed=None,rand_seed=None):
    env = make(Config.GAME_TYPE, num_env)
    if Config.FRAME_STACK > 1:
        env = VecFrameStack(env, Config.FRAME_STACK)

    if Config.EPSILON_GREEDY > 0:
        env = wrappers.EpsilonGreedyWrapper(env, Config.EPSILON_GREEDY)

    if Config.MU_OP == 1:
        env = wrappers.RandSeedWrapper(env,Config.INI_LEVELS,rand_seed)
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
    set_global_seeds(seed * 100 + rank)

    # For wandb package to visualize results curves
    config = Config.get_args_dict()
    config['global_seed'] = seed
    wandb.init(project="coinrun",notes=" generative adversarial train",tags=["try"],config=config)

    utils.setup_mpi_gpus()
    utils.mpi_print('Set up gpu')
    utils.mpi_print(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # nenvs is how many envs run parallel on a cpu
    # VenEnv class allows parallel rollout
    # nenvs = Config.NUM_ENVS
    total_timesteps = int(128*10**6)
    phase_timesteps = int(total_timesteps // Config.TRAIN_ITER)
    eval_timesteps = 0

    with tf.Session(config=config):
        sess = tf.get_default_session()

        # init env
        nenv = Config.NUM_ENVS
        env = make_general_env(nenv,rand_seed=seed)
        utils.mpi_print('Set up env')

        policy = policies_back.get_policy()
        utils.mpi_print('Set up policy')

        seed_set_log = []

        for t in range(Config.TRAIN_ITER):
        # ============ GARL =================
            # 1. optimize
            _, act_model = learn_func(sess=sess,policy=policy,env=env,
                                      log_interval=args.log_interval,
                                      save_interval=args.save_interval,
                                      nsteps=Config.NUM_STEPS,
                                      nminibatches=Config.NUM_MINIBATCHES,
                                      lam=Config.GAE_LAMBDA,gamma=Config.GAMMA,
                                      noptepochs=Config.PPO_EPOCHS,
                                      ent_coef=Config.ENTROPY_COEFF,
                                      vf_coef=Config.VF_COEFF,
                                      max_grad_norm=Config.MAX_GRAD_NORM,
                                      lr=lambda f : f * Config.LEARNING_RATE,
                                      cliprange=lambda f : f * Config.CLIP_RANGE,
                                      total_timesteps = phase_timesteps,index = t)

            # 2. mutate
            last_set = list(env.get_seed_set())
            last_set_res = eval_set(sess,nenv, last_set,rep_count=3)
            eval_timesteps += last_set_res['eval_steps']
            utils.mpi_print("eval_steps",eval_timesteps)
            utils.mpi_print("gen : "+str(t-1))
            mpi_print_res(last_set_res)

            for k in last_set:
                # version1: all seed replace
                #if last_set_res['mean'] == 10.0:
                rs = env.replace_seed(k)
            utils.mpi_print("mutate all seed")

            # 3. evaluate
            curr_set = list(env.get_seed_set())
            curr_set_res = eval_set(sess,nenv,curr_set,rep_count=3,save=True,idx = t)
            eval_timesteps += curr_set_res['eval_steps']
            utils.mpi_print("eval_steps",eval_timesteps)
            utils.mpi_print("gen :"+str(t))
            mpi_print_res(curr_set_res)

            # 4. replace
            assert len(last_set) == len(curr_set), "currset should same size with lastset"
            next_set = []
            replace_count = 0
            for idxs in range(len(last_set)):
                last_fit = last_set_res['mean'][idxs]
                curr_fit = curr_set_res['mean'][idxs]
                if last_fit > curr_fit:
                    # score decrease means diffculty increase
                    # replace
                    replace_count += 1
                    next_set.append(curr_set[idxs])
                else:
                    next_set.append(last_set[idxs])
            seed_set_log.append(next_set)
            env.set_ini_set(next_set)
            utils.mpi_print("replace ",replace_count)
            utils.mpi_print("set new seed",next_set)

            res = eval_set(sess, nenv, None, rep_count=1000, save=True, idx=t)
            utils.mpi_print("PPO train timesteps",t * phase_timesteps)
            utils.mpi_print("GARL test timesteps",eval_timesteps)
            mpi_print_res(res)
            wandb.log({
                'Steps_elapsed':t * phase_timesteps,
                'Test_mean':res['mean'],
                'Test_max':res['max'],
                'Test_norm':res['nor']
            })

        res = eval_set(sess, nenv, None, rep_count=1000, save=True, idx=t)
        utils.mpi_print("PPO train timesteps",t * phase_timesteps)
        utils.mpi_print("GARL test timesteps",eval_timesteps)
        mpi_print_res(res)

    env.close()

if __name__ == '__main__':
    main()

