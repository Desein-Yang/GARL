#============================================
# Load an agent trained with train_agent.py
# and eval it on specific set
#
#
# @author:yangqi
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
from garl.ppo2_v4 import load_model,save_model
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

def eval_test(sess,nenvs,train_set=None,eval_seed=0,train=False,
              is_high=False,idx=None,rep_count=1000,log=True,render=True):
    """evaluate for rep_count times in given eval_set
    not every level is evaluated for same times"""
    if train_set is None and train is False: # test eval
        rng = np.random.RandomState(eval_seed)
        # test
        train_set = list(rng.randint(0,2**31-1,rep_count))
        Config.HIGH_DIFFICULTY = is_high
        Config.NUM_LEVELS = 0

    env  = make_eval_env(nenvs,train_set)
    agent = create_act_model(sess,env,nenvs)

    sess.run(tf.global_variables_initializer())
    if type(idx) is int:
        is_loaded = load_model(sess,base_name=str(idx)+'M')
    else:
        is_loaded = load_model(sess,base_name=idx)

    if is_loaded is False:
        print("NO PARAMS LOADED!")
        return 0.0

    eval_log, eval_steps = eval_set(sess, nenvs, train_set, 1)
    scores = list(eval_log.values())
    if log:
        if not train:
            mpi_print("--------------Test set eval------------")
        else:
            mpi_print("--------------Train set eval------------")
        mpi_print("rep count",rep_count)
        mpi_print("load path",is_loaded)
        mpi_print("rew mean",np.sum(scores)/rep_count)
        mpi_print("succ rate",np.sum(scores==10.0)/rep_count)
        mpi_print("-----------------------------------------")

        if render:
            env.export("test.mp4")
    return eval_log


def eval_set(sess,nenvs,seed_set,rep_count=1):
    """evaluate every seed for rep_count times in given seed_set
    use model sav_runid_0
    return scores in sequence"""
    batch_size = len(seed_set)
    assert nenvs > rep_count, "repeat times shoud less than nenvs"

    env1 = make_eval_env(nenvs, 0)
    rep_each = int(rep_count // rep_count)
    agent = create_act_model(sess,env1,nenvs)

    # key is seed, value is perf
    eval_log = {}
    eval_steps = 0

    res_seed = list(seed_set)
    batch_size = len(res_seed)
    #print("garl/eval/eval_set/seed_set ",seed_set)
    eval_times = 0

    def should_eval():
        if len(res_seed) == 0:
            return False
        return True

    while(should_eval()):
        env1.set_seed(set(res_seed))
        env1,seeds,scores,steps = eval(sess, agent, env1, rep_each)
        #print("garl/eval/eval_set/scores ",scores)
        #print("garl/eval/eval_set/seeds ",seeds)

        for seed,score in zip(seeds,scores):
            if seed in res_seed:
                eval_log[str(seed)] = score
                res_seed.remove(seed)

        #print("garl/eval/eval_set/res_seeds ",res_seed)
        eval_steps += np.sum(steps)
        eval_times += len(scores)

        seed_set_ = [str(s) for s in seed_set]

    print("garl/eval/eval_set/eval_log ",eval_log)
    print("garl/eval/eval_set/eval_times ",eval_times)
    return eval_log, eval_steps


def eval(sess,agent,env,rep_count=1):
    """evaluate with env(seed has been set) for rep_count times
    return scores"""

    nenvs = env.num_envs
    obs = env.reset()

    seeds,scores,steps = [],[],[]
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        #return np.sum(score_counts) < rep_count * nenvs
        return len(scores) < rep_count * nenvs

    state = agent.initial_state
    done = np.zeros(nenvs)

    #print('garl/eval/eval/env.get_seed',env.get_seed())
    #print('garl/eval/eval/should_continue',should_continue())
    while should_continue():
        action, values, state, _ = agent.step(obs, state, done)
        obs, rew, done, epiinfo = env.step(action)
        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                # discard
                #if score_counts[i] < rep_count:
                #    score_counts[i] += 1
                #    if 'episode' in info[i]:
                #        scores[i] += info[i].get('episode')['r']
                #        steps[i] += info[i].get('episode')['l']
                if len(scores) < rep_count * nenvs:
                    if 'episode' in epiinfo[i]:
                        scores.append(epiinfo[i].get('episode')['r'])
                        steps.append(epiinfo[i].get('episode')['l'])
                        seeds.append(epiinfo[i].get('episode')['s'])

        if done[0]:
            curr_rews[:] = 0

    return env, seeds, scores, steps
