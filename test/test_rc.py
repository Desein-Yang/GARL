from garl.wrappers import RandConvWrapper
from garl.coinrunenv import make
from garl.config import Config
import numpy as np
from PIL import Image
import tensorflow as tf


def make_general_env(num_env,seed=None,rand_seed=None):
    env = make('standard', num_env)
    env = RandConvWrapper(env)

    return env

if __name__ == '__main__':
    from garl.setup_utils import setup_and_load
    args = setup_and_load()
    print("setup")
    env = make_general_env(1,rand_seed=123)
    print("finish")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    act = np.array([1])
    obs = env.step_clean(act)[0]
    obs2 = env.step(act)[0]
    import pdb;pdb.set_trace()

    im = Image.fromarray(obs2.numpy())
    im.save('ob2.jpeg')
    im = Image.fromarray(obs.numpy())
    im.save('ob.jpeg')
    import pdb;pdb.set_trace()



