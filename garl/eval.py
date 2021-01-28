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
            env = wrappers.RandSeedWrapper(env,[seed],1)
        elif type(seed) == list:
            env = wrappers.RandSeedWrapper(env,seed,len(seed))
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

def eval_test(sess,nenvs,eval_set=None,eval_seed=0,train=False,
              is_high=False,idx=None,rep_count=1000,log=True):
    """evaluate for rep_count times in given eval_set
    not every level is evaluated for same times"""
    if eval_set is None and train is False: # test eval
        rng = np.random.RandomState(eval_seed)
        eval_set = list(rng.randint(0,2**31-1,rep_count))
        Config.HIGH_DIFFICULTY = is_high
        Config.NUM_LEVELS = 0
    env  = make_eval_env(nenvs,eval_set)
    agent = create_act_model(sess,env,nenvs)
    rep_each = int(rep_count // nenvs) + 1

    sess.run(tf.global_variables_initializer())
    if idx is None:
        is_loaded = load_model(sess,base_name=None)
    else:
        is_loaded = load_model(sess,base_name=str(i)+'M')

    if is_loaded is False:
        print("NO PARAMS LOADED!")
        return 0.0

    scores, steps = eval(sess, agent, env, rep_each)
    scores = scores[:rep_count]
    steps = steps[:rep_count]

    if log:
        if not train:
            mpi_print("--------------Test set eval------------")
        else:
            mpi_print("--------------Train set eval------------")
        mpi_print("rep count",rep_count)
        mpi_print("load path",is_loaded)
        mpi_print("eval seed",eval_seed)
        mpi_print("rew mean",np.sum(scores)/rep_count)
        mpi_print("succ rate",np.sum(scores==10.0)/rep_count)
        mpi_print("eval step",np.sum(steps))
        mpi_print("-----------------------------------------")

    return scores, steps, eval_set


def eval_set(sess,nenvs,seed_set,rep_count=1,log=True):
    """evaluate every seed for rep_count times in given seed_set
    use model sav_runid_0
    return scores in sequence"""
    batch_size = len(seed_set)
    assert nenvs > rep_count, "repeat times shoud less than nenvs"

    env = make_eval_env(nenvs, 0)
    rep_each = int(rep_count // rep_count)
    agent = create_act_model(sess,env,nenvs)

    sess.run(tf.global_variables_initializer())
    load_model(sess,base_name=None)

    mean_scores = np.zeros((batch_size,))
    eval_steps = 0

    for i,seed in enumerate(seed_set):
        env.set_seed(set([seed]))
        scores,steps = eval(sess, agent, env, rep_each)

        # we only use 0-rep_count score in Vec envs
        scores = scores[:rep_count]
        eval_steps += np.sum(steps)
        mean_scores[i] = np.mean(scores)

        if log:
            mpi_print("seed",env.get_seed(),"score",mean_scores[i])

    return mean_scores, eval_steps

def eval(sess,agent,env,rep_count=1):
    """evaluate with env(seed has been set) for rep_count times
    return scores"""

    nenvs = env.num_envs
    obs = env.reset()

    scores,steps = [],[]
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        #return np.sum(score_counts) < rep_count * nenvs
        return len(scores) < rep_count * nenvs

    state = agent.initial_state
    done = np.zeros(nenvs)

    while should_continue():
        action, values, state, _ = agent.step(obs, state, done)
        obs, rew, done, epiinfo = env.step(action)
        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                #if score_counts[i] < rep_count:
                #    score_counts[i] += 1
                #    if 'episode' in info[i]:
                #        scores[i] += info[i].get('episode')['r']
                #        steps[i] += info[i].get('episode')['l']
                if len(scores) < rep_count * nenvs:
                    if 'episode' in epiinfo[i]:
                        scores.append(epiinfo[i].get('episode')['r'])
                        steps.append(epiinfo[i].get('episode')['l'])
        if done[0]:
            curr_rews[:] = 0

    return np.array(scores),np.array(steps)

#if __name__ == '__main__':
#    main()
