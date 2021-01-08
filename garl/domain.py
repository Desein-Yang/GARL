# ===========================================
# this is for domain randomization for env
# and other utils
#
# copyright QiYANG
# ===========================================
import tensorflow as tf
import os
import joblib
import numpy as np
from mpi4py import MPI
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from garl.config import Config
from garl import setup_utils, wrappers

import platform
import pickle
import coinrunenv

def setup_and_load(cmd=True, **kwargs):
    args = Config.initialize_args(use_cmd_line_args=cmd,**kwargs)

    return args

def make_general_env(num_env, seed=0, use_sub_proc=True):
    env = coinrunenv.make(Config.GAME_TYPE, num_env)

    if Config.FRAME_STACK > 1:
        env = VecFrameStack(env, Config.FRAME_STACK)

    epsilon = Config.EPSILON_GREEDY

    if epsilon > 0:
        env = wrappers.EpsilonGreedyWrapper(env, epsilon)

    return env


def add_mutate_wrapper():
    """mutation and crossover on env"""
    if Config.MU_OP == 0:
        pass
    elif Config.MU_OP == 1:
        pass
    else:
        pass


def init_args_and_thread():
    pass


def init_level_seed():
    pass


def init_themes():
    pass


def diff_func():
    """To calculate difference of two env"""
    if Config.DIFF_OP == 0:
        # use perf as diff
        pass
    elif Config.DIFF_OP == 1:
        # use value
        pass
    elif Config.DIFF_OP == 2:
        # ...
        pass
    return diff

if __name__ == '__main__':
    main()
