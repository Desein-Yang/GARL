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
from garl.eval import eval_test
from garl import setup_utils

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

def enjoy_env_sess(sess,i=None,test_seed=100000,wandb_log=False):
    should_render = False
    should_eval = True
    #should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP
    nenv = Config.NUM_ENVS
    Config.RUN_ID = Config.RESTORE_ID
    succ_rate = 0

    #if should_eval:
    #    #env = utils.make_general_env(Config.NUM_EVAL)
    #    env = make_eval_env(Config.NUM_EVAL,seed=[0])
    #    should_render = False
    #else:
    #    env = utils.make_general_env(1)
        #env = wrappers.add_final_wrappers(env)
    #hashtable = {}

    #mpi_print('getseed',env.get_seed())

    if should_render:
        from gym.envs.classic_control import rendering

    #nenvs = env.num_envs

    #agent = create_act_model(sess, env, nenvs)

    #sess.run(tf.global_variables_initializer())

    # load model
    #if i is None:
    #    loaded_params = utils.load_params_for_scope(sess, 'model')
    #elif i == 'Best':
    #    base_name = 'Best'
    #    load_path = Config.get_load_filename(
    #        restore_id=Config.RESTORE_ID,
    #        base_name=base_name
    #    )
    #    if os.path.exists(Config.LOGDIR + load_path):
    #        loaded_params = utils.load_params_for_scope(sess,'model',
    #                                        load_path=load_path,
    #                                        load_key='default')
    #    else:
    #        loaded_params = False
    #elif i == 0:
    #    should_eval = False
    #    mean_score, succ_score = 0.0,0.0
    #    loaded_params = True
    #else:
    #    base_name = str(i)+'M'
    #    load_path = Config.get_load_filename(
    #        restore_id=Config.RESTORE_ID,
    #        base_name=base_name
    #    )
    #    print(Config.LOGDIR + load_path)
    #    if os.path.exists(Config.LOGDIR + load_path):
    #        loaded_params = utils.load_params_for_scope(sess,'model',
    #                                        load_path=load_path,
    #                                        load_key='default')
    #    else:
    #        loaded_params = False
    #if not loaded_params:
    #    print('NO SAVED PARAMS LOADED')
    #    should_eval = False

    # eval
    # obs = env.reset()
    # t_step = 0

    if should_render:
        #viewer = rendering.SimpleImageViewer()
        obs = []

    should_render_obs = not Config.IS_HIGH_RES

    def maybe_render(info=None):
        if should_render and not should_render_obs:
            env.render()

    maybe_render()

    # === origin ======================
    # scores = np.array([0] * nenvs)
    # episodes = np.array([0] * nenvs)
    # score_counts = np.array([0] * nenvs)
    # curr_rews = np.zeros((nenvs, 3))

    # def should_continue():
    #    if should_eval:
    #        return np.sum(score_counts) < rep_count * nenvs
    #
    #    return False

    #state = agent.initial_state
    #done = np.zeros(nenvs)
    #obs = []

    #while should_continue():
    #    action, values, state, _ = agent.step(obs, state, done)
    #    obs, rew, done, infos = env.step(action)

    #    if should_render and should_render_obs:
    #        if np.shape(obs)[-1] % 3 == 0:
    #            ob_frame = obs[0,:,:,-3:]
    #        else:
    #            ob_frame = obs[0,:,:,-1]
    #            ob_frame = np.stack([ob_frame] * 3, axis=2)
            #viewer.imshow(ob_frame)
            #im = Image.fromarray(ob_frame)
            #im.save("screen.jpg")

    #    curr_rews[:,0] += rew

    #    for i, d in enumerate(done):
    #        if d:
    #            if score_counts[i] < rep_count:
    #                score_counts[i] += 1

    #                if 'episode' in infos[i]:
    #                    if (infos[i].get('episode')['r']==10.0):
    #                        succ_rate += 1
    #                    scores[i] += infos[i].get('episode')['r']
    #                    episodes[i] += infos[i].get('episode')['l']
    #                    if 's' in infos[i].get('episode').keys():
    #                        seed = infos[i].get('episode')['s']
    #                        hashtable[seed] = infos[i].get('episode')['r']

        #if t_step % 100 == 0:
            #mpi_print('t', t_step, values[0], done[0], rew[0], curr_rews[0], np.shape(obs))

    #    maybe_render(infos[0])

    #   t_step += 1

    #    if should_render:
    #        time.sleep(.02)

    #   if done[0]:
    #        if should_render:
    #            mpi_print('ep_rew', curr_rews)

    #       curr_rews[:] = 0

    #result = 0

    if should_eval is True:
        final_test = {}

        if Config.TRAIN_EVAL:
            mpi_print('-----------Train-set----------------------')
            opt_hist = joblib.load(Config.LOGDIR+'opt_hist')
            train_set = list(opt_hist["hist"][-1])
            eval_log = eval_test(sess, nenv, train_set=train_set,
                            train=True,idx=i,
                            is_high=False, rep_count=1000, log=False)


        elif Config.TEST_EVAL:
            mpi_print('-----------Test-set----------------------')
            mpi_print('test random seed(np.randomstate)',test_seed )
            eval_log = eval_test(sess, nenv, train_set=None, train=False,
                                 eval_seed=test_seed,is_high=True,
                                 idx=i, rep_count=1000, log=False)

        final_test['performance'] = eval_log
        scores = list(eval_log.values())
        mean_score = np.mean(scores)
        final_test['mean_score'] = mean_score
        joblib.dump(final_test,setup_utils.file_to_path("final_test"))

        mpi_print('mean_score', mean_score)
        mpi_mean_score = utils.mpi_average([mean_score])
        mpi_print('mpi_mean', mpi_mean_score)

        if wandb_log:
            wandb.log({
                'Step_elapsed':i * 8 * 1e6,
                'Rew_mean':mean_score,
            })


def main():
    setup_utils.setup_and_load()
    wandb_log = True
    if wandb_log:
        wandb.init(
            project="coinrun",
            name=Config.RESTORE_ID+'test',
            config=Config.get_args_dict()
        )
    with tf.Session() as sess:
        for i in range(0,256,8):
            enjoy_env_sess(sess,i,wandb_log)
        # i = 8
        # enjoy_env_sess(sess,None,False)
        # print("test Best model")
        # enjoy_env_sess(sess,'Best',False)

if __name__ == '__main__':
    main()
