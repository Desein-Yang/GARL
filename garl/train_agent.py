"""
geenraltive adversarial reinforcement learning
"""
import time
from mpi4py import MPI
import tensorflow as tf
import wandb
from baselines.common import set_global_seeds
import main_utils as utils
import setup_utils, policies_back, wrappers, ppo2
from config import Config
import domain as domain

def learn_func(**kwargs):
    if Config.USE_EVO == 1:
        f = ppo2.learn(**kwargs)
    else:
        # origin ppo
        f = ppo2.learn(**kwargs)
    return f

def main():
    args = setup_utils.setup_and_load()
    setup_utils.load_for_setup_if_necessary()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('size',size)
     
    # For wandb package to visualize results curves
    config = Config.get_args_dict()
    wandb.init(
        project="coinrun",
        notes=" generative adversarial train",
        tags=["try"],
        config=config
    )
    
    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()
    utils.mpi_print('Set up gpu')
    utils.mpi_print(args)
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # nenvs is how many envs run parallel on a cpu
    # VenEnv class allows parallel rollout
    # nenvs = Config.NUM_ENVS
    total_timesteps = int(256*10**6)
    phase_timesteps = total_timesteps / Config.TRAIN_ITER
    #env = utils.make_general_env(nenvs, seed=rank)
    #utils.mpi_print('Set up env')

    with tf.Session(config=config):
        for t in range(T_max):
        # ============ GARL =================

            # 1. initialize new envs (mutate)
            envs_set = []
            new_envs_set = []
            group = int(Config.INI_LEVELS // Config.NUM_ENVS)
            for i in range(group):
                nenvs = Config.NUM_ENVS
                domain.init_args_and_thread()
                domain.init_level_seed()
                domain.init_themes()
                env = make_general_env(nenvs, seed=rank)
                #if np.random.randn() < Config.MU_RATE:
                new_env = domain.add_mutate_wrapper(env)
                
           # 2. evaluate 
                env = wrappers.add_final_wrappers(env)
        
                policy = policies_back.get_policy()
	        result = evaluate(sess,model,env,rep_count)
                f = result['succ_rate']
                result_new = evaluate(sess,model,env,rep_count)
                f_new = result_new['succ_rate']

                d = diff(envs_set)
                d_new = diff(envs_set)

           # 3. mixed (select)
           child_fit = f + d * Config.NC_COEF
           maxid = np.argmax(child_fit)
           child_env = envs_set[maxid]
           if child_fit > father_fit:
               env = child_env
           else:
               env = env
        
           #4. user envs
        # =========== optimizer ==================
            
           learn_func(policy=policy,
                    env=env,
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
                    total_timesteps=phase_timesteps)

if __name__ == '__main__':
    main()

