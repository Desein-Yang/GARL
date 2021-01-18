"""
Load an agent trained with train_agent.py and
"""

import time
import wandb
import tensorflow as tf
import numpy as np
from coinrun import setup_utils
import coinrun.main_utils as utils
from coinrun.config import Config
from coinrun import policies, wrappers

mpi_print = utils.mpi_print

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def load_model(sess, base_name=None, i=0):
    if base_name is None:
        base_name = str(i)
    else:
        base_name = base_name + "_"+str(i)
    filename = Config.get_save_file(base_name)
    utils.load_params_for_scope(sess, 'model', load_path = filename, load_key='default')


def enjoy_env_sess(sess,step,base_name,idx):
    should_render = False
    should_eval = True
    #should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP

    if should_eval:
        env = utils.make_general_env(Config.NUM_EVAL)
        should_render = False
    else:
        env = utils.make_general_env(1)

    env = wrappers.add_final_wrappers(env)

    if should_render:
        from gym.envs.classic_control import rendering

    nenvs = env.num_envs

    agent = create_act_model(sess, env, nenvs)


    #load_file = setup_utils.restore_file(
    #    Config.RESTORE_ID,
    #    base_name=base_name
    #)
    sess.run(tf.global_variables_initializer())
    if 'GA' not in Config.RESTORE_ID:
        loaded_params = utils.load_params_for_scope(sess, 'model')
    else:
        load_model(sess,base_name,idx)

    obs = env.reset()
    t_step = 0

    if should_render:
        viewer = rendering.SimpleImageViewer()

    should_render_obs = not Config.IS_HIGH_RES

    def maybe_render(info=None):
        if should_render and not should_render_obs:
            env.render()

    maybe_render()

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

        if should_render and should_render_obs:
            if np.shape(obs)[-1] % 3 == 0:
                ob_frame = obs[0,:,:,-3:]
            else:
                ob_frame = obs[0,:,:,-1]
                ob_frame = np.stack([ob_frame] * 3, axis=2)
            viewer.imshow(ob_frame)

        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1

                    if 'episode' in info[i]:
                        scores[i] += info[i].get('episode')['r']

        #if t_step % 100 == 0:
            #mpi_print('t', t_step, values[0], done[0], rew[0], curr_rews[0], np.shape(obs))

        maybe_render(info[0])

        t_step += 1

        if should_render:
            time.sleep(.02)

        if done[0]:
            if should_render:
                mpi_print('ep_rew', curr_rews)

            curr_rews[:] = 0

    result = 0

    if should_eval:
        mean_score = np.mean(scores) / rep_count
        max_idx = np.argmax(scores)
        mpi_print('scores', scores / rep_count)
        mpi_print('mean_score', mean_score)
        mpi_print('max idx', max_idx)

        mpi_mean_score = utils.mpi_average([mean_score])
        mpi_print('mpi_mean', mpi_mean_score)
        succ_rate = np.mean(scores==10.0) / rep_count
        step_elapsed = step * 10**6
        #wandb.log({
        #    'Rew_mean':mean_score,
        #    'Succ_rate':succ_rate,
        #    'Step_elapsed':step_elapsed
        #})

    return mean_score, succ_rate, step_elapsed

def main():
    utils.setup_mpi_gpus()
    #setup_utils.load_for_setup_if_necessary()
    setup_utils.setup_and_load()
    wandb.init(
        project="coinrun",config=Config.get_args_dict()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        path = './logs/'+Config.RESTORE_ID.replace('_','-')
        bases, steps, outid = test_list(path)
        print(bases,steps)
        means, succs, step_es = [],[],[]
        for i in range(0,len(bases)-1):
            mean, succ, step = enjoy_env_sess(sess,steps[i],bases[i],outid[i])
            means.append(mean)
            succs.append(succ)
            step_es.append(step)
        means = two_sort(step_es, means)
        succs = two_sort(step_es, succs)
        step_es = two_sort(step_es,step_es)
        for i in range(len(step_es)):
            wandb.log({
                'Rew_mean':means,
                'Succ_rate':succ,
                'Step_elapsed':step_es
            })

def two_sort(a,b):
    assert len(a) == len(b)
    ind = np.argsort(np.array(a))
    resa = np.zeros_like(np.array(a))
    resb = np.zeros_like(np.array(b))
    for i,rank in enumerate(ind):
        print(rank)
        print(resa,resb)
        resa[i] = a[rank]
        resb[i] = b[rank]
    return resa,resb

def test_list(path):
    import os
    files,steps,outids = [],[],[]
    for filename in os.listdir(path):
        if 'sav' in filename:
            if 'GA' in filename:
                if 'M' not in filename.split('_')[4]:
                    outind = int(filename.split('_')[4])
                    base_name = str(outind)
                    inind = 0
                else:
                    outind = int(filename.split('_')[5])
                    inind = int(filename.split('_')[4][:-1])
                    base_name = filename.split('_')[4]+'_'+str(outind)
                steps.append((16*outind+inind))
                files.append(base_name)
                outids.append(outind)
            else:
                if filename.split('_')[3] == '0':
                    return None, 256
                inind = int(filename.split('_')[3][:-1])
                steps.append(inind*8)
                base_name = str(8*inind)+'M'

    return files, steps, outids

if __name__ == '__main__':
    main()
