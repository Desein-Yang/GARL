import numpy as np
from garl.main_utils import mpi_print
from garl.eval import eval_set
import wandb,joblib

class TaskOptimizer(object):
    def __init__(self,env,rep=3,eval_limit=1e6,log=True):
        self.env = env
        self.iter = 0
        self.is_log = log
        self.rep = rep
        self.eval_steps = 0
        self.eval_limit = eval_limit
        self.hist = []

        self.size = 0
        self.train_set_size = 0

    def gen(self):
        pass

    def select(self):
        pass

    def calFit(self):
        pass

    def run(self):
        pass

    def reset(self):
        self.hist = []
        self.train_set_size = 0
        self.iter = 0
        self.eval_steps = 0


class SeedOptimizer(TaskOptimizer):
    def __init__(self,logdir,env,rand_seed=None,rep=3,
                 eval_limit=1e6,train_set_limit=500,
                 spare_size=10000,ini_size=100,log=True):
        TaskOptimizer.__init__(self, env)
        #super(TaskOptimizer, self).__init__(env)
        self.rep = rep
        self.eval_limit = eval_limit

        self.spare_size = spare_size
        self.ini_size = ini_size
        self.rs = np.random.RandomState(rand_seed)
        seed_set = self.rs.randint(0,2**31-1,self.spare_size + self.ini_size)
        self.spare_set = set(seed_set[:self.spare_size])
        self.ini_set = set(seed_set[self.spare_size:])

        # all used seed set
        self.train_set_hist = set(seed_set)
        self.train_set_size = self.size
        self.train_set_limit = train_set_limit

        # If use diversity, phi = 0
        self.phi = 0
        self.iter = 0
        self.step_elapsed = 0
        self.train_rew = 0
        self.if_log = log
        self.logdir = logdir
        self.log()

    def calDiv(self,p,vec):
        assert type(p) is float
        vec1 = np.ones_like(vec) * p

        return np.sqrt(np.sum(np.square(vec1 - vec2)))

    def calFit(self,vec1):
        if self.phi != 0:
            div1 = np.zeros_like(vec1)
            for i in range(vec1.shape[0]):
                div1[i] = self.calDiv(vec1[i],vec1)
        else:
            div1 = 0.0

        fit1 = vec1 + self.phi * div1
        return fit1

    def gen(self,seed_set):
        for i,seed in enumerate(seed_set):
            b = np.random.choice(list(self.spare_set))
            if b not in seed_set:
                seed_set[i] = b
        return seed_set

    def replace(self,sess):
        last_set = list(self.env.get_seed())
        last_set_rew = self.eval(sess,last_set,self.rep)

        curr_set = self.gen(last_set)
        curr_set_rew = self.eval(sess,curr_set,self.rep)

        next_set = []
        last_fit = self.calFit(last_set_rew)
        curr_fit = self.calFit(curr_set_rew)

        for idxs in range(len(last_set)):
            if last_fit[idxs] > curr_fit[idxs]:
                # score decrease means diffculty increase
                next_set.append(curr_set[idxs])
            else:
                next_set.append(last_set[idxs])
        self.train_set_hist.union(set(next_set))
        self.train_set_size = len(self.train_set_hist)
        self.step_elapsed += self.eval_steps
        self.log()

        return next_set

    def select(self,sess,ratio=0.5):
        """select 50% best"""
        last_set = list(self.env.get_seed())
        last_set_rew = self.eval(sess,last_set,self.rep)

        curr_set = self.gen(last_set)
        curr_set_rew = self.eval(sess,curr_set,self.rep)

        next_set = []
        last_fit = self.calFit(last_set_rew)
        curr_fit = self.calFit(curr_set_rew)

        fits = np.concatenate((curr_fit,last_fit),0)
        curr_set.extend(last_set)

        # [1,2,3..] ascend
        rank_fit = np.argsort(fits)
        for idxs, pos in enumerate(rank_fit):
            # to get rank 1 index of last_set
            next_set.append(curr_set[pos])
            if idxs > ratio * len(set(curr_set)):
                break

        self.train_set_hist = self.train_set_hist.union(set(next_set))
        self.train_set_size = len(self.train_set_hist)

        self.step_elapsed += self.eval_steps
        self.log()
        return next_set

    def add(self,sess,ratio=0.5):
        """If new level is  more difficult than average difficulty, add it
        until add 50% * now set size"""
        last_set = list(self.env.get_seed())
        curr_set = self.gen(last_set)
        curr_set_rew = self.eval(sess,curr_set,self.rep)
        curr_fit = self.calFit(curr_set_rew)

        next_set = last_set.copy()
        count = 0

        for idx,fit in enumerate(curr_fit):
            if fit < self.train_rew:
                next_set.append(curr_set[idx])
                count += 1
            if count > ratio * len(last_set):
                break

        self.train_set_hist = next_set
        if len(next_set) > self.train_set_limit:
            next_set = next_set[:self.train_set_limit]
        self.train_set_size = len(self.train_set_hist)
        self.step_elapsed += self.eval_steps
        self.log()

        return next_set

    def log(self):
        if self.if_log:
            wandb.log({
                'step_elapsed':self.step_elapsed,
                'iter':self.iter,
                'eval_steps':self.eval_steps,
                'train_set_size':self.train_set_size,
            })

        idx = int(self.step_elapsed // 1e6)
        joblib.dump(self.hist, self.logdir + "opt_hist"+str(idx)+'M')

    def eval(self,sess,env_set,rep):
        nenv = self.env.num_envs
        scores, steps = eval_set(sess,nenv,env_set,rep_count=rep)

        self.eval_steps +=  np.sum(steps)

        return scores

    def run(self,sess,env,step_elapsed,train_rew,mode='add'):
        self.env = env
        self.eval_steps = 0
        self.step_elapsed = step_elapsed
        self.train_rew = train_rew

        if self.train_set_size >= self.train_set_limit:
            should_continue = False
        else:
            should_continue = True
        # Optimizen until eval limit reach
        while(should_continue):
            if mode == 'replace':
                next_set = self.replace(sess)
            elif mode == 'select':
                next_set = self.select(sess,ratio=0.5)
            elif mode =='add':
                next_set = self.add(sess,ratio=0.2)
                should_continue = False
            else:
                raise ValueError

            self.env.set_seed(next_set)

            if (self.eval_steps > self.eval_limit
            or self.train_set_size >= self.train_set_limit):
                should_continue = False

        # Output nextset to be trained on
        self.env.set_seed(next_set)
        self.hist.append(next_set)
        mpi_print("set new seed",next_set)
        self.iter += 1

        return self.env, self.step_elapsed
