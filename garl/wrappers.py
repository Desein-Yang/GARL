import gym
import numpy as np
import tensorflow as tf

class EpsilonGreedyWrapper(gym.Wrapper):
    """
    Wrapper to perform a random action each step instead of the requested action,
    with the provided probability.
    """
    def __init__(self, env, prob=0.05):
        gym.Wrapper.__init__(self, env)
        self.prob = prob
        self.num_envs = env.num_envs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform()<self.prob:
            action = np.random.randint(self.env.action_space.n, size=self.num_envs)
        return self.env.step(action)


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {'render.modes': []}
        env.reward_range = (-float('inf'), float('inf'))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                info = infos[0]
                if 'aux_rew' in info:
                    self.num_aux_rews = len(infos[0]['aux_rew'])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)
                self.long_aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i,:] += info['aux_rew']
                    self.long_aux_rewards[i,:] += info['aux_rew']

            for i, d in enumerate(done):
                if d:
                    epinfo = {
                        'r': round(self.rewards[i], 6),
                        'l': self.lengths[i],
                        't': 0,
                        # Hash Table: GARL modified
                        #'s': self.get_cur_seed()[i]
                    }
                    aux_dict = {}

                    for nr in range(self.num_aux_rews):
                        aux_dict['aux_' + str(nr)] = self.aux_rewards[i,nr]

                    if 'ale.lives' in infos[i]:
                        game_over_rew = np.nan

                        is_game_over = infos[i]['ale.lives'] == 0

                        if is_game_over:
                            game_over_rew = self.long_aux_rewards[i,0]
                            self.long_aux_rewards[i,:] = 0

                        aux_dict['game_over_rew'] = game_over_rew

                    epinfo['aux_dict'] = aux_dict

                    infos[i]['episode'] = epinfo

                    self.rewards[i] = 0
                    self.lengths[i] = 0
                    self.aux_rewards[i,:] = 0

            return obs, rew, done, infos

        self.reset = reset
        self.step = step

# ========= self defined =================

def add_mutate_wrappers(env):
    """Mu_op is 1, use another seed in extra set.
       Mu_op is 2, use gauss noise on physical."""
    if Config.MU_OP == 0:
        return env
    elif Config.MU_OP == 1:
        return RandSeedWrapper(env)
    elif Config.MU_OP == 2:
        return ParamWrapper(env)
    elif Config.MU_OP == 3:
        return RandConvWrapper(env)

class ParamWrapper(gym.Wrapper):
    def __init__(self,env,args,seed):
        super(ParamWrapper, self).__init__(env)
        self.params = args
        self.mu = np.zeros_like(args)
        self.sigma = np.ones_like(args)
        self.rs = np.random.RandomState(seed)
        self.step = step

    def mutate(self):
        self.params += np.around(self.rs.randn(len(args)),decimals=1)
        args = {
            'gravity':self.params[0],
            'air_control':self.params[1],
            'max_jump':self.params[2],
            'max_speed':self.params[3],
            'mix_rate':self.params[4],
        }
        self.env.set_phys(args)
        self.env.reset()

    def reset():
        pass

    def get_param(self):
        return self.params

# provide modifiable seed API
class RandSeedWrapper(gym.Wrapper):
    def __init__(self,env,ini_set=None,ini_size=100):
        super(RandSeedWrapper, self).__init__(env)
        self.env = env
        if ini_set is None:
            ini_set = np.random.randint(0,2**31-1,ini_size)
        self.ini_set = ini_set
        self.cur_set = ini_set
        self.env.set_seed(self.ini_set)
        self.hist = set(ini_set)

    def replace_seed(self,a,b):
        """replace a with b"""
        if a in self.cur_set:
            self.cur_set.remove(a)
        self.cur_set.add(b)
        self.hist = self.hist.union(set(b))

        self.set_seed(self.cur_set)
        return b

    def add_seed(self,a):
        if type(a) is int:
            b = [a]
        self.cur_set.union(set(b))
        self.hist = self.hist.union(set(a))
        self.env.set_seed(self.cur_set)

    def set_seed(self,seed):
        if seed is None:
            seed = []
        elif type(seed) is int:
            seed = [seed]
        elif type(seed) is set:
            seed = list(seed)
        self.env.set_seed(seed)

    def get_seed(self):
        return self.env.get_seed()

    def reset_seed(self):
        self.hist = set(self.ini_set)
        self.cur_set = self.ini_set
        self.set_seed(self.ini_set)

class HashTableWrapper(gym.Wrapper):
    def __init__(self,env):
        """Add a hashtable to record rews and steps of specific seed.
        Should use with (and after) RandSeed and EpisodeReward Wrapper"""
        pass

    def get_hash(self,seed):
        pass

    def add_hash(self,seed):
        pass


# back 20200128
class RandSeed2Wrapper(gym.Wrapper):
    def __init__(self,env,ini_size,spare_size,rand_seed):
        super(RandSeedWrapper, self).__init__(env)
        if rand_seed is not None:
            self.rs = np.random.RandomState(rand_seed)
        else:
            self.rs = np.random.RandomState()
        self.set_size = spare_size
        if self.set_size > 1:
            self.mutate_set = set(self.rs.randint(0,2**31-1,self.set_size))
        self.hist = []
        self.ini_size = ini_size
        self.ini_set = set(self.rs.randint(0,2**31-1,self.ini_size))
        self.set_seed(self.ini_set)

    def replace_seed(self,a):
        """mutate levels and remove seed a
        add another seed b from mutate_set"""
        self.ini_set.remove(a)
        b = np.random.choice(list(self.mutate_set))
        #self.mutate_set.remove(b)
        self.ini_set.add(b)
        self.hist.append(self.ini_set)

        self.set_seed(self.ini_set)
        return b

    def add_seed(self,a):
        if type(a) is int:
            b = [a]
        self.ini_set.union(set(b))
        self.set_seed(self.ini_set)

    def set_ini_set(self,ini_set):
        self.ini_set = set(ini_set)
        self.set_seed(self.ini_set)

    def set_seed(self,seed):
        self.env.set_seed(seed)
        self.env.reset()

    def get_seed_set(self):
        return self.ini_set

    def get_seed(self):
        return self.env.get_seed()

    def reset_seed(self):
        self.ini_set = set(self.rs.randint(0,2**31-1,self.ini_size))
        self.mutate_set = self.mutate_set.union(set(self.hist))
        self.hist = []
        self.set_seed(self.ini_set)


class RandConvWrapper(gym.Wrapper):
    def __init__(self,env,kernel=3,depth=3,rh=0.2,channel=3,seed=None):
        super(RandConvWrapper, self).__init__(env)
        self.env = env
        self.kernel = kernel
        self.depth = depth
        self.rh = rh
        self.channel = channel

        fan_in = channel * kernel * kernel
        fan_out = depth * kernel * kernel

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed
        self.initializer = tf.contrib.layers.xavier_initializer(
                    uniform=True,
                    seed=self.seed,
                    dtype=tf.float32
        )
        #self.initializer = tf.initialiers.he_normal(seed)

    def reset(self):
        obs = self.env.reset()
        self.initializer = tf.contrib.layers.xavier_initializer(
                    uniform=True,
                    seed=self.seed,
                    dtype=tf.float32
            )

        return self.conv(obs)

    def step(self, act):
        obs,rew,done,epi = self.env.step(act)
        out = self.conv(obs)
        with tf.Session() as sess:
            obs = sess.run(out)

        return obs,rew,done,epi

    def step_clean(self,act):
        obs,rew,done,epi = self.env.step(act)

        return obs,rew,done,epi

    def set_seed(self,seed):
        self.seed = seed
        self.initializer = tf.contrib.layers.xavier_initializer(
                    uniform=True,
                    seed=self.seed,
                    dtype=tf.float32
        )

    def get_seed(self,seed):
        return self.seed

    def conv(self,images):
        images = tf.to_float(images)
        mask_vbox = tf.Variable(tf.zeros_like(images, dtype=bool), trainable=False)
        mask_shape = tf.shape(images)

        mh = tf.cast(
                tf.cast(mask_shape[1], dtype=tf.float32)*self.rh, dtype=tf.int32
        )
        mw = mh * 2
        mask_vbox = mask_vbox[:,:mh,:mw].assign(
            tf.ones([mask_shape[0],mh,mw,mask_shape[3]],dtype=bool)
        )

        img = tf.where(mask_vbox, x=tf.zeros_like(images), y= images)
        img = tf.to_float(img)
        rand_img = tf.layers.conv2d(img, self.depth, self.kernel,
                                    padding='same',
                                    kernel_initializer=self.initializer,
                                    trainable=False,
                                    name='randcnn')
        out = tf.where(mask_vbox,x=images,y=rand_img,name='randout')
        return out

    def save():
        import joblib
        return self.conv.weight()


# =======================================
def add_final_wrappers(env):
    env = EpisodeRewardWrapper(env)

    return env
