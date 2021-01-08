"""
Load an agent trained with train_agent.py and 
"""
import joblib,time,os
import tensorflow as tf
import wandb
import numpy as np
import setup_utils
import main_utils as utils
from config import Config
import policies,wrappers
from ppo2 import Model
from mpi4py import MPI

mpi_print = utils.mpi_print


def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
    #model = Model(policy=act,ob_space=ob_space,ac_space=ac_space)
    return act

def evaluate(sess,model,env,rep_count=20):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    
    #env = utils.make_general_env(1)
    env = wrappers.add_final_wrappers(env)
    nenvs = env.num_envs
    
    agent = create_act_model(sess, env, nenvs)

    #sess.run(tf.global_variables_initializier())
    #loaded_params = utils.load_params_for_scope(sess,'model')

    #if os.path.exists(load_path):
    #    model.load(load_path)
    #else:
    #    print('NO SAVED PARAMS LOADED')

    obs = env.reset()
    t_step = 0

    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return True

    state = agent.initial_state
    done = np.zeros(nenvs)

    while should_continue():
        action, values, state, _ = agent.step(obs, state, done)
        obs, rew, done, info = env.step(action)
        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1
                if 'episode' in info[i]:
                    scores[i] += info[i].get('episode')['r']

        t_step += 1
        if done[0]:
            curr_rews[:] = 0

    result = {
        'steps_elapsed':steps_elapsed,
    }

    if should_eval:
        testset_size = rep_count * nenvs
        max_score = np.max(scores)
        mean_score = np.sum(scores) / testset_size
        succ_rate = np.sum(scores==10.0) / testset_size
        if size > 1:
            mean_score = utils.mpi_average([mean_score])
        result['scores'] = scores
        result['testset_size'] = testset_size
        result['mean_score'] = mean_score
        result['max_score'] = max_score
        result['succ_rate'] = succ_rate

    return result

def main():
    utils.setup_mpi_gpus()
    setup_utils.setup_and_load()
    env = utils.make_general_env(1)
    load_path = './logs/DO-DQN-1-123/sav_DO_DQN_1_123_0'
    with tf.Session() as sess:
        result = test(sess,load_path,env,20)
    print(result)

if __name__ == '__main__':
    main()
