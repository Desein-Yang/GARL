#============================================
# Load an agent trained with train_agent.py
# and eval it on specific set
#
#
# copyright QiYANG
# ===========================================

import joblib,time,os
import tensorflow as tf
from mpi4py import MPI
import wandb
import numpy as np

import garl.main_utils as utils
from garl.config import Config
from garl import setup_utils,policies_back,wrappers
from garl.coinrunenv import make
from garl.ppo2 import load_model,save_model
mpi_print = utils.mpi_print

def make_eval_env(num_env,seed=None):
    env = make(Config.GAME_TYPE, num_env)
    if Config.FRAME_STACK > 1:
        env = wrappers.VecFrameStack(env, Config.FRAME_STACK)

    if seed is not None:
        if type(seed) == int:
            env = wrappers.RandSeedWrapper(env,1,None)
            env.set_ini_set(set([seed]))
        elif type(seed) == list:
            env = wrappers.RandSeedWrapper(env,len(seed),None)
            env.set_ini_set(set(seed))
        else:
            raise ValueError
    env = wrappers.EpisodeRewardWrapper(env)
    return env

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies_back.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=tf.AUTO_REUSE)
    return act

def eval_set(sess,nenvs, seed_set=None,rep_count=1,save=False,idx=0):
    if seed_set is None:
        # test
        rep_each = int(rep_count // nenvs) + 1
        seed_set = list(np.random.randint(0,2**31-1,rep_count))
        env =  make_eval_env(nenvs, seed_set)
        agent = create_act_model(sess,env,nenvs)

        sess.run(tf.global_variables_initializer())
        load_model(sess,base_name=None,i=idx)

        scores, steps = eval(sess, agent, env, rep_each)
        scores = scores[:rep_count]

        res = {}
        res['eval_steps'] = np.sum(steps)
        res['seed_set'] = seed_set
        res['mean'] = np.sum(scores) / rep_count
        res['std'] = np.std(scores)
        res['succ'] = np.sum(scores==np.max(scores)) / rep_count
        res['max'] = np.max(scores)
        res['min'] = np.min(scores)
        if res['max'] > 0:
            res['nor'] = res['mean'] / res['max']
        else:
            res['nor'] = 0.0

    else:
        batch_size = len(seed_set)
        assert nenvs > rep_count, "repeat times shoud less than nenvs"
        env = make_eval_env(nenvs, 0)
        rep_each = int(rep_count // rep_count)
        agent = create_act_model(sess,env,nenvs)

        sess.run(tf.global_variables_initializer())
        load_model(sess,base_name=None,i=idx)

        mean_score = np.zeros((batch_size,))
        succ_score = np.zeros((batch_size,))
        max_score = np.zeros((batch_size,))
        min_score = np.zeros((batch_size,))
        std_score = np.zeros((batch_size,))
        nor_score = np.zeros((batch_size,))
        eval_steps = 0

        for i,seed in enumerate(seed_set):
            env.set_ini_set(set([seed]))
            scores,steps = eval(sess, agent, env, rep_each)

            # we only use 0-rep_count score in Vec envs
            scores = scores[:rep_count]
            mean_score[i] = np.sum(scores) / rep_count
            max_score[i] = np.max(scores)
            min_score[i] = np.min(scores)
            std_score[i] = np.std(scores)
            succ_score[i] = np.sum(scores==np.max(scores)) / rep_count
            eval_steps += np.sum(steps)
            if max_score[i] > 0:
                nor_score[i] = min_score[i] / max_score[i]
            else:
                nor_score[i] = 0

        res = {
            'eval_steps':eval_steps,
            'seed_set':seed_set,
            'mean':mean_score,
            'max':max_score,
            'min':min_score,
            'succ':succ_score,
            'std':std_score,
            'nor':nor_score
        }

    if save:
        import joblib
        joblib.dump(res, Config.LOGDIR+"eval_set_"+str(idx))

    return res

def eval(sess,agent,env,rep_count=1):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    nenvs = env.num_envs
    #rep_count = int(rep_count / nenvs)
    #agent = create_act_model(sess, env, nenvs)
    obs = env.reset()
    t_step = 0

    scores = np.array([0] * nenvs)
    steps = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        return np.sum(score_counts) < rep_count * nenvs

    state = agent.initial_state
    done = np.zeros(nenvs)

    while should_continue():
        action, values, state, _ = agent.step(obs, state, done)
        obs, rew, done, info = env.step(action)
        curr_rews[:,0] += rew
        t_step += 1

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1

                    if 'episode' in info[i]:
                        scores[i] = info[i].get('episode')['r']
                        steps[i] = info[i].get('episode')['l']
        if done[0]:
            #mpi_print('ep_rew',curr_rews)
            curr_rews[:] = 0

    return scores,steps

#if __name__ == '__main__':
#    main()
