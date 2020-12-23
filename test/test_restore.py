import wandb
import numpy as np
#import coinrun.enjoy as test 
import coinrun.main_utils as utils
from coinrun.config import Config
import coinrun.setup_utils as setup_utils
import tensorflow as tf
from mpi4py import MPI
import coinrun.wrappers as wrappers
import coinrun.policies_back as policies

mpi_print = utils.mpi_print

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    agent = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
    return agent

def enjoy_env_sess(sess,checkpoint,overlap):
    #base_name = str(8*checkpoint)  + 'M'
    #load_file = setup_utils.restore_file(Config.RESTORE_ID,base_name=base_name)
    should_eval = True
    mpi_print('test levels seed',Config.SET_SEED)
    mpi_print('test levels ',Config.NUM_LEVELS)
    rep_count = 50
    
    env = utils.make_general_env(20)
    env = wrappers.add_final_wrappers(env)
    nenvs = env.num_envs

    sess.run(tf.global_variables_initializer())
    args_now = Config.get_args_dict()
    #args_run = utils.load_args()  
    agent = create_act_model(sess, env, nenvs)

    # load name is specified by config.RESTORE_ID adn return True/False
    if checkpoint != 32:
        base_name = str(8*checkpoint)  + 'M'
    elif checkpoint == 0:
        mean_score = 0.0
        succ_rate = 0.0 
        wandb.log({
            'Rew_mean':mean_score,
            'Succ_rate':succ_rate,
            'Step_elapsed':steps_elapsed
        })
        return mean_score, succ_rate
    else:
        base_name = None
   
    # env init here
    load_file = setup_utils.restore_file(
                Config.RESTORE_ID,
                overlap_config=overlap,
                base_name=base_name
    )
    
    is_loaded = utils.load_params_for_scope(sess, 'model')
    if not is_loaded:
        mpi_print('NO SAVED PARAMS LOADED')
        return mean_score, succ_rate

    obs = env.reset()
    t_step = 0
    
    scores = np.zeros((nenvs,rep_count))
    eplens = np.zeros((nenvs,rep_count))
    #scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    # curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return True

    state = agent.initial_state
    done = np.zeros(nenvs)

    def rollout(obs,state,done):
        """rollout for rep * nenv times and return scores"""
        t = 0
        count = 0
        rews = np.zeros((nenvs, rep_count))
        while should_continue():
            action, values, state, _ = agent.step(obs, state, done)
            obs, rew, done, info = env.step(action)
            rews[:,count] += rew
            t += 1
            
            for i, d in enumerate(done):
                if d:
                    eplens[i][count]=t
                    if score_counts[i] < rep_count:
                        score_counts[i] += 1
                        count = score_counts[i] - 1
                        # aux score
                        if 'episode' in info[i]:
                            scores[i][count] = info[i].get('episode')['r']

        return scores, rews, eplens
   
    if is_loaded:
        mpi_print(load_file)	
        scores, rews, eplens = rollout(obs,state,done)
     
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if size == 1:
        if rank == 0:
            testset_size = rep_count * nenvs
            utils.save_pickle(scores,Config.LOGDIR+'scores')
            mean_score = np.sum(scores) / testset_size
            succ_rate = np.sum(scores==10.0) / testset_size
            mpi_print('cpus ',size)
            mpi_print('testset size',testset_size)
            # NUM_LEVELS = 0 means unbounded set so the set size is rep_counts * nenvs
            # each one has a new seed(maybe counted)
            # mpi_print('score detail',scores.flatten())
            mpi_print('succ_rate',succ_rate)
            steps_elapsed = checkpoint * 8000000
            mpi_print('steps_elapsed:',steps_elapsed)
            mpi_print('mean score',mean_score)
            wandb.log({
                       'Rew_mean':mean_score,
                       'Succ_rate':succ_rate,
                       'Step_elapsed':steps_elapsed
            })
            #mpi_print('mean score of each env',[np.mean(s) for s in scores])
    else:
        testset_size = rep_count * nenvs
        succ = np.sum(scores=10.0)/ testset_size
        succ_rate = utils.mpi_average([succ])
        mean_score_tmp = np.sum(scores) / testset_size
        mean_score = utils.mpi_average([mean_score_tmp])
        if rank == 0:
            mpi_print('testset size',rep_count * nenvs * size)
            mpi_print('load file name',load_file)
            mpi_print('testset size',testset_size)
            # NUM_LEVELS = 0 means unbounded set so the set size is rep_counts * nenvs
            # each one has a new seed(maybe counted)
            # mpi_print('score detail',scores.flatten())
            mpi_print('succ_rate',succ_rate)
            mpi_print('mean score',mean_score)
            wandb.log({
                       'Rew_mean':mean_score,
                       'Succ_rate':succ_rate
            })
    
    return mean_score,succ_rate 
 
def main():
    # load from restore file
    args_dict = utils.load_args()
    # train args of restore id
    test_args = setup_utils.setup_and_load()
    if 'NR' in Config.RESTORE_ID:
        Config.USE_LSTM = 2
    if 'dropout' in Config.RESTORE_ID: 
        Config.DROPOUT = 0
        Config.USE_BATCH_NORM = 0
    
    wandb.init(
        project="coinrun",
        notes="test",
        tags=["baseline","test"],
        config=Config.get_args_dict()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    seed = np.random.randint(100000)
    Config.SET_SEED = seed
   
    overlap = {
        'set_seed':Config.SET_SEED,
        'rep':Config.REP,
        'num_levels':Config.NUM_LEVELS,
        'use_lstm':Config.USE_LSTM,
        'dropout':Config.DROPOUT,
        'use_batch_norm':Config.USE_BATCH_NORM
    } 
   
    load_file = Config.get_load_filename(restore_id=Config.RESTORE_ID) 
    mpi_print('load file name',load_file)
    mpi_print('seed',seed)
    mpi_print("---------------------------------------") 
    for checkpoint in range(1,33):
        with tf.Session() as sess:
            steps_elapsed = checkpoint * 8000000
            mpi_print('steps_elapsed:',steps_elapsed)
            enjoy_env_sess(sess,checkpoint,overlap)

if __name__ == '__main__':
    import traceback
    try:
        main()
    except:
        print(traceback.format_exc())

