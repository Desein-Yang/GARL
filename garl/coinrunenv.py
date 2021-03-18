#===========================================================
# Python interface to the CoinRun shared library using ctypes.
# On import, this will attempt to build the shared library.
#
# Modification:
# 1. init_args() add random seed
# 2. init_args() add random params(optional)
# copyright QiYANG
#===========================================================G
import os
import atexit
import random
import sys
from ctypes import c_int, c_char_p, c_float, c_bool

import gym
import gym.spaces
import numpy as np
import numpy.ctypeslib as npct
import garl
from baselines.common.vec_env import VecEnv
from baselines import logger

from garl.config import Config
from mpi4py import MPI
from baselines.common import mpi_util

# if the environment is crashing, try using the debug build to get
# a readable stack trace
DEBUG = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

game_versions = {
    'standard':   1000,
    'platform': 1001,
    'maze': 1002,
}

def build():
    lrank, _lsize = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
    if lrank == 0:
        dirname = os.path.dirname(__file__)
        if len(dirname):
            make_cmd = "QT_SELECT=5 make -C %s" % dirname
        else:
            make_cmd = "QT_SELECT=5 make"

        r = os.system(make_cmd)
        if r != 0:
            logger.error('coinrun: make failed')
            sys.exit(1)
    MPI.COMM_WORLD.barrier()

build()

if DEBUG:
    lib_path = '.build-debug/coinrun_cpp_d'
else:
    lib_path = '.build-release/coinrun_cpp'

# https://python3-cookbook.readthedocs.io/zh_CN/latest/c15/p01_access_ccode_using_ctypes.html
lib = npct.load_library(lib_path, os.path.dirname(__file__))
lib.init.argtypes = [c_int]
lib.get_NUM_ACTIONS.restype = c_int
lib.get_RES_W.restype = c_int
lib.get_RES_H.restype = c_int
lib.get_VIDEORES.restype = c_int
lib.get_NUM_LEVELS.restype = c_int
lib.get_LEVEL_SEED.restype = c_int
lib.get_LEVEL_SEEDS.restype = c_int
#lib.get_SEED_SEQ.restype = c_int

lib.vec_create.argtypes = [
   c_int,    # game_type
    c_int,    # nenvs
    c_int,    # lump_n
    c_bool,   # want_hires_render
    c_float,  # default_zoom
    ]
lib.vec_create.restype = c_int

lib.vec_close.argtypes = [c_int]
lib.vec_game_over.argtypes = [c_int]

lib.vec_step_async_discrete.argtypes = [c_int, npct.ndpointer(dtype=np.int32, ndim=1)]

lib.initialize_args.argtypes = [npct.ndpointer(dtype=np.int32, ndim=1)]
lib.initialize_set_seeds.argtypes = [
    npct.ndpointer(dtype=np.int32, ndim=1),
    npct.ndpointer(dtype=np.int32, ndim=1),
    c_int
]
lib.initialize_diff.argtypes = [c_int]
lib.initialize_phys.argtypes = [npct.ndpointer(dtype=np.float, ndim=1)]
lib.initialize_set_monitor_dir.argtypes = [c_char_p, c_int]

lib.vec_wait.argtypes = [
    c_int,
    npct.ndpointer(dtype=np.uint8, ndim=4),    # normal rgb
    npct.ndpointer(dtype=np.uint8, ndim=4),    # larger rgb for render()
    npct.ndpointer(dtype=np.float32, ndim=1),  # rew
    npct.ndpointer(dtype=np.bool, ndim=1),     # done
    npct.ndpointer(dtype=np.int32, ndim=1),     # seed
    ]

already_inited = False

def init_args_and_threads(cpu_count=1,
                          monitor_csv_policy='all',
			  rand_seed=None):
    """
    Perform one-time global init for the CoinRun library.  This must be called
    before creating an instance of CoinRunVecEnv.  You should not
    call this multiple times from the same process.
    You should generate level seeds set to init env.
    """
    os.environ['COINRUN_RESOURCES_PATH'] = os.path.join(SCRIPT_DIR, 'assets')
    is_high_difficulty = Config.HIGH_DIFFICULTY

    if rand_seed is None:
        rand_seed = random.SystemRandom().randint(0, 1000000000)

        # ensure different MPI processes get different seeds (just in case SystemRandom implementation is poor)
        mpi_rank, mpi_size = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
        rand_seed = rand_seed - rand_seed % mpi_size + mpi_rank

    int_args = np.array([
		int(is_high_difficulty),
		Config.INI_LEVELS,
		int(Config.PAINT_VEL_INFO),
		Config.USE_DATA_AUGMENTATION,
		game_versions[Config.GAME_TYPE],
		Config.SET_SEED,
                rand_seed,
                False, # if use specified diff?
                False  # if use specified physical params?
		]).astype(np.int32)

    lib.initialize_args(int_args)
    lib.initialize_set_monitor_dir(logger.get_dir().encode('utf-8'), {'off': 0, 'first_env': 1, 'all': 2}[monitor_csv_policy])

    global already_inited
    if already_inited:
        return

    # init thread and asset
    lib.init(cpu_count)
    already_inited = True

def initialize_physical(args,diff=True,phys=True):
    """Init physical paramters in coinrun
       Args: dict
    """
    if diff:
        float_args = np.array([
		args['gravity'],
                args['air_control'],
                args['max_jump'],
                args['max_speed'],
                args['mix_rate']
        ]).astype(np.float)
        lib.initialize_phys(float_args)

    if phys:
        int_args = np.array([args["diffculty"]]).astype(np.int)
        lib.initialize_diff(int_args)

def initialize_seed(level_seed=None,w=None):
    """Init seed of each level
       Should be before vec_create()
       All env in Vec shared a same seed.
       if weight is not None use w to weighted sample.
    """
    assert level_seed is not None
    if type(level_seed) is int:
        level_seed = [level_seed]

    if len(level_seed) == 0:
        assert Config.SET_SEED != -1,"set seed should not be -1"
        rs = np.random.RandomState(Config.SET_SEED)
        level_seed = rs.randint(0,2**31-1,Config.INI_LEVELS)

    if w is not None:
        assert len(level_seed) == len(w)
    else:
        w = np.ones_like(level_seed)

    level_seed = np.array(level_seed).astype(np.int32)
    w = np.array(w).astype(np.int32)

    print("initialize seed:level_seed",level_seed)
    print("initialize weight:w",w)
    lib.initialize_set_seeds(level_seed, w, len(level_seed))


def initialize_theme():
    pass

@atexit.register
def shutdown():
    global already_inited
    if not already_inited:
        return
    lib.coinrun_shutdown()

class CoinRunVecEnv(VecEnv):
    """
    This is the CoinRun VecEnv, all CoinRun environments are just instances
    of this class with different values for `game_type`

    `game_type`: int game type corresponding to the game type to create, see `enum GameType` in `coinrun.cpp`
    `num_envs`: number of environments to create in this VecEnv
    `lump_n`: only used when the environment creates `monitor.csv` files
    `default_zoom`: controls how much of the level the agent can see
    """
    def __init__(self, game_type, num_envs, seed, lump_n=0, default_zoom=5.0):
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))

        self.NUM_ACTIONS = lib.get_NUM_ACTIONS()
        self.RES_W       = lib.get_RES_W()
        self.RES_H       = lib.get_RES_H()
        self.VIDEORES    = lib.get_VIDEORES()

        self.buf_rew = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_seed = np.zeros([num_envs], dtype=np.int32)
        self.buf_rgb   = np.zeros([num_envs, self.RES_H, self.RES_W, 3], dtype=np.uint8)
        self.hires_render = Config.IS_HIGH_RES
        if self.hires_render:
            self.buf_render_rgb = np.zeros([num_envs, self.VIDEORES, self.VIDEORES, 3], dtype=np.uint8)
        else:
            self.buf_render_rgb = np.zeros([1, 1, 1, 1], dtype=np.uint8)

        num_channels = 1 if Config.USE_BLACK_WHITE else 3
        obs_space = gym.spaces.Box(0, 255, shape=[self.RES_H, self.RES_W, num_channels], dtype=np.uint8)

        super().__init__(
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=gym.spaces.Discrete(self.NUM_ACTIONS),
            )
        if seed is None:
            seed = []
        self.default_zoom = default_zoom
        self.game_type = game_type
        self.lump_n = lump_n
        self.num_envs = num_envs
        initialize_seed(seed)
        self.handle = lib.vec_create(
            game_versions[game_type],
            self.num_envs,
            self.lump_n,
            self.hires_render,
            default_zoom)
        self.dummy_info = [{} for _ in range(num_envs)]

    def set_seed(self,seed,w=None,num_envs=None):
        """set global variables LEVEL_SEED or LEVEL_SEEDS in cpp"""
        initialize_seed(seed,w)
        self.reset()

    def get_num_levels(self):
        return lib.get_NUM_LEVELS()

    def get_seed(self):
        """get current level seed-set in cpp"""
        num_levels = self.get_num_levels()

        seed,w= [0] * num_levels,[0] * num_levels
        for i in range(num_levels):
            seed[i] = lib.get_LEVEL_SEEDS(i)
            w[i] = lib.get_LEVEL_WEIGHTS(i)
        return seed, w

    def get_cur_seed(self):
        return self.buf_seed

    # delete
    def get_cur_seed_delete(self):
        """get seed sequence of a vector level"""
        seed = [0] * self.num_envs
        #for i in range(self.num_envs):
        #    seed[i] = lib.get_SEED_SEQ(i)

        return seed

    def get_phys(self):
        return lib.get_PHYC_PARAM()

    def get_diff(self):
        return lib.get_DIFFCULTY()

    def set_phys(self,args):
        initialize_physical(args,True,True)

    def __del__(self):
        if hasattr(self, 'handle'):
            lib.vec_close(self.handle)
        self.handle = 0

    def close(self):
        lib.vec_close(self.handle)
        self.handle = 0

    def reset(self):
        lib.vec_game_over(self.handle)
        #print("CoinRun ignores resets")
        obs, _, _, _ = self.step_wait()
        return obs

    def get_images(self):
        if self.hires_render:
            return self.buf_render_rgb
        else:
            return self.buf_rgb

    # step = step_async(actions), return step_wait()
    def step_async(self, actions):
        assert actions.dtype in [np.int32, np.int64]
        actions = actions.astype(np.int32)
        lib.vec_step_async_discrete(self.handle, actions)

    def step_wait(self):
        self.buf_rew = np.zeros_like(self.buf_rew)
        self.buf_done = np.zeros_like(self.buf_done)
        # GARL
        self.buf_seed = np.zeros_like(self.buf_seed)

        lib.vec_wait(
            self.handle,
            self.buf_rgb,
            self.buf_render_rgb,
            self.buf_rew,
            self.buf_done,
            self.buf_seed)

        obs_frames = self.buf_rgb
        #print("bufrew",self.buf_rew)
        #print("bufdone",self.buf_done)
        print("bufseed",self.buf_seed)

        if Config.USE_BLACK_WHITE:
            obs_frames = np.mean(obs_frames, axis=-1).astype(np.uint8)[...,None]

        return obs_frames, self.buf_rew, self.buf_done, self.dummy_info

def make(env_id, num_envs, seed=None, **kwargs):
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CoinRunVecEnv(env_id, num_envs, seed, **kwargs)

def setup_and_load(use_cmd = True, **kwargs):
    args = Config.initialize_args(use_cmd=True, **kwargs)
    init_args_and_threads(4)
    return args

if __name__ == '__main__':
    import garl.main_utils as utils
    import garl.setup_utils
    utils.setup_mpi_gpus()
    setup_and_load()
    from garl.wrappers import RandSeedWrapper
#    args = {
#       'gravity':0.2,
#        'air_control':0.15,
#        'max_jump':1.5,
#        'max_speed':0.5,
#        'mix_rate':0.2,
#        'diffculty':2
#    }
    seeds = [123,453,345,567,678]
    env = make('standard', 3, seeds)
    env = RandSeedWrapper(env,seeds,len(seeds))
    print('seed',env.get_seed())
    seeds = [123,453,345,567,678,890]
    env.set_seed(seeds)
    print('seed2',env.get_seed())
    env.set_seed(465)
    print('seed3',env.get_seed())
    print('seed4',env.get_cur_seed())
    act = np.array([env.action_space.sample()])
    env.step(act)

