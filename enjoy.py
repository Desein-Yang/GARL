"""
Load an agent trained with train_agent.py and
"""
import os
import time
import wandb
import tensorflow as tf
import numpy as np
from garl import setup_utils
import garl.main_utils as utils
#from coinrun.config import Config
from garl.config import Config
from garl import  wrappers
import garl.policies_back as policies
mpi_print = utils.mpi_print
import joblib

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def make_eval_env(num_env,seed=None):
    from garl.coinrunenv import make
    env = make(Config.GAME_TYPE, num_env)
    if Config.FRAME_STACK > 1:
        env = wrappers.VecFrameStack(env, Config.FRAME_STACK)

    env = wrappers.EpisodeRewardWrapper(env)
    if seed is not None:
        env.set_seed(seed)
    return env

def enjoy_env_sess(sess,i=None,wandb_log=False):
    should_render = False
    should_eval = True
    #should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP
    succ_rate = 0

    if Config.TRAIN_EVAL:
        opt_hist = joblib.load(Config.LOGDIR+'opt_hist')
        #eval_set = [int(i) for i in a.split('  ')]
        train_set = opt_hist[-1]

    if Config.TEST_EVAL:
        train_set = np.random.randint(0,2**31-1,1000)
    mpi_print('trainset',train_set)
    if should_eval:
        #env = utils.make_general_env(Config.NUM_EVAL)
        env = make_eval_env(Config.NUM_EVAL,seed=train_set)
        should_render = False
    else:
        env = utils.make_general_env(1)
    #env = wrappers.add_final_wrappers(env)
    mpi_print('getseed',env.get_seed())

    if should_render:
        from gym.envs.classic_control import rendering

    nenvs = env.num_envs

    agent = create_act_model(sess, env, nenvs)

    sess.run(tf.global_variables_initializer())
    if i is None:
        loaded_params = utils.load_params_for_scope(sess, 'model')
    elif i == 0:
        should_eval = False
        mean_score, succ_score = 0.0,0.0
        loaded_params = True
    else:
        base_name = str(i)+'M'
        load_path = Config.get_load_filename(
            restore_id=Config.RESTORE_ID,
            base_name=base_name
        )
        print(Config.LOGDIR + load_path)
        if os.path.exists(Config.LOGDIR + load_path):
            loaded_params = utils.load_params_for_scope(sess,'model',
                                            load_path=load_path,
                                            load_key='default')
        else:
            loaded_params = False
    if not loaded_params:
        print('NO SAVED PARAMS LOADED')
        should_eval = False

    #should_eval=False
    #last_set = [1159135105,920906118,159803024,230190482,
    #            903463848,344031272,2104313899,1580380463,
    #            2006247609,1933271611,643366587,841327420,
    #            737905224,2089619659,883940690,1768682713,
    #            730734435,1454788835,1021694966,1823563002]

    #scores, steps = eval_set(sess,20,last_set,
    #                          rep_count=1)
    #mpi_print("rew mean",scores)
    #mpi_print("rew mean",np.mean(scores))

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
    episodes = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return False

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
                        if (info[i].get('episode')['r']==10.0):
                            succ_rate += 1
                        scores[i] += info[i].get('episode')['r']
                        episodes[i] += info[i].get('episode')['l']

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

    if should_eval is True:
        mean_score = np.mean(scores) / rep_count
        succ_score = succ_rate / (rep_count * nenvs)
        max_idx = np.argmax(scores)
        if Config.TRAIN_EVAL:
            mpi_print('-----------Train-set----------------------')
        elif Config.TEST_EVAL:
            mpi_print('-----------Test-set----------------------')
        mpi_print('scores', scores / rep_count)
        mpi_print('mean_score', mean_score)
        mpi_print('succ_score', succ_score)
        mpi_print('max idx', max_idx)

        mpi_mean_score = utils.mpi_average([mean_score])
        mpi_print('mpi_mean', mpi_mean_score)

        result = mean_score
        mpi_print('mpimean',mpi_mean_score)

        if wandb_log:
            wandb.log({
                'Step_elapsed':i * 8 * 1e6,
                'Rew_mean':mean_score,
                'Succ_rate':succ_score
            })

    return result

def main():
    setup_utils.setup_and_load()
    wandb_log = True
    if wandb_log:
        wandb.init(
            project="coinrun",
            name=Config.RESTORE_ID+'test',
            config=Config.get_args_dict())
    with tf.Session() as sess:
        for i in range(0,256,8):
        #i = 8
            enjoy_env_sess(sess,i,wandb_log)
        enjoy_env_sess(sess,None,False)

if __name__ == '__main__':
    main()
