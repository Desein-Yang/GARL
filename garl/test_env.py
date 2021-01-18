from garl.coinrunenv import make
import garl.wrappers as wrappers
from garl.train_agent import setup_and_load
from garl.config import Config
def make_general_env(num_env,seed=None,rand_seed=None):
    env = make(Config.GAME_TYPE, num_env)

    env = wrappers.RandSeedWrapper(env,Config.INI_LEVELS,rand_seed)

    env = wrappers.EpisodeRewardWrapper(env)
    return env


setup_and_load()
env = make_general_env(1,1)
s = env.get_seed()
print(s)
