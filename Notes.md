# CoinRun env 

Coinrun 环境主要为了评估强化学习中的agent泛化能力而提出，更准确地说是 gym 平台为基准定制的一个游戏任务。底层由 C++ 实现并接入 gym 平台。

文件结构：
coinrun.cpp : 提供底层支持  
coinrunenv.py : 调用 cpp 库创建游戏环境类
    - CoinrunVecEnv()
    - make(env_id,nums) 创建环境
    - build()
    - init_args_and_threads() Coinrun 整个库的初始化
interactive.py : 提供交互式桌面环境
config.py : 可以修改环境参数配置
    - ConfigSingle()

用于测试和训练
- ppo2.py & train_agent.py 训练 PPO
- enjoy.py 人工玩
- random_agent.py & test_coinrun.py 测试环境文件
- wrapper.py episilon greedy 封装
- tb_utils 用于tensorflowboard 可视化
- main_utils 各种小应用工具
- policies.py 定制各种结构

## game

game_versions = {
    'standard':   1000,
    'platform': 1001,
    'maze': 1002,
}

用的就是coinrun, maze 是为了仿照 zhang2018
env_id 的三个选择

## Config

可以配置的参数

```python
python -m coinrun.train_agent []
// train algo
[runid] 保存文件名 tmp
[resid] 加载设置名 None
[gamet] 游戏模式，standard,platform,maze
[arch]  网络结构，nature impala impalalarge
[lstm]  是否包含网络结构 0
[ne]    并行环境数 32
[nlev]  训练集关卡数量 0(2^32unbounded)
[si]    保存间隔 10 
[set-seed] 训练集关卡的随机种子数 -1 不用随机种子
[train-eval] 从训练集采样测试集
[test-eval]  从非训练集采样测试集
[test]   是否拿一半worker出来测试
// PPO
[ns]    总帧数(M)
[nmb]   minibatch 数量 8
[ppoeps] epochs 3
[ent]   entropy coefficient 0.01
[lr]    learning rate 5e-4
[gamma] gamma 0.999
// game
[pvi]   游戏agent速度是否显示在观察中 -1
[norm]  是否 batch norm 0
[dropout]是否dropout 0
[uda]   是否 data augmentation 0
[l2]    l2 weight 0.0
[eps]   ep greedy prob 0
[fs]    framestack 跳帧 1
[ubw]   是否灰度
[highd] 是否高难度
[hres]  是否高对比度
// evaluation
[num-eval] 评估环境线程数
[rep]   每个环境评估的episode数

```
## 使用指南


```python
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config
nenvs = Config.NUM_ENVS
env = utils.make_general_env(nenvs, seed=rank)
env = wrappers.add_final_wrappers(env)

def make_general_env(num_env, seed=0, use_sub_proc=True):
    from coinrun import coinrunenv
    env = coinrunenv.make(Config.GAME_TYPE, num_env)
    if Config.FRAME_STACK > 1:
        env = VecFrameStack(env, Config.FRAME_STACK)
    epsilon = Config.EPSILON_GREEDY
    if epsilon > 0:
        env = wrappers.EpsilonGreedyWrapper(env, epsilon)
    return env
    
def make(env_id, num_envs, **kwargs):
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CoinRunVecEnv(env_id, num_envs, **kwargs)

```
## 游戏参数

C++ 中可以改的参数

Maze
```cpp
int spawnpos[2];
int w, h;
int game_type;
int* walls;
int coins;
bool is_terminated;

float gravity;
float max_jump;
float air_control;
float max_dy;
float max_dx;
float default_zoom;
float max_speed;
float mix_rate;
void init_physics() {
    if (game_type == CoinRunMaze_v0) {
        default_zoom = 7.5;
    } else {
        default_zoom = 5.0;
    }
        
    gravity = .2;
    air_control = .15;

    max_jump = 1.5;
    max_speed = .5;
    mix_rate = .2;

    max_dy = max_jump * max_jump / (2*gravity);
    max_dx = max_speed * 2 * max_jump / gravity;
}
```
Agent 
```cpp
int theme_n;
float x, y, vx, vy;
float spring = 0;
float zoom = 1.0;
float target_zoom = 1.0;
uint8_t render_buf[RES_W*RES_H*4];
uint8_t* render_hires_buf = 0;
bool game_over = false;
float reward = 0;
float reward_sum = 0;
bool is_facing_right;
bool ladder_mode;
int action_dx = 0;
int action_dy = 0;
int time_alive;
bool support;
FILE *monitor_csv = 0;
double t0;
```
Initialization
```cpp
    USE_HIGH_DIF = int_args[0] == 1;
    NUM_LEVELS = int_args[1];
    PAINT_VEL_INFO = int_args[2] == 1;
    USE_DATA_AUGMENTATION = int_args[3] == 1;
    DEFAULT_GAME_TYPE = int_args[4];

    int training_sets_seed = int_args[5];
    int rand_seed = int_args[6];

    if (NUM_LEVELS > 0 && (training_sets_seed != -1)) {
    global_rand_gen.seed(training_sets_seed);

    USE_LEVEL_SET = true;

    LEVEL_SEEDS = new int[NUM_LEVELS];

    for (int i = 0; i < NUM_LEVELS; i++) {
        LEVEL_SEEDS[i] = global_rand_gen.randint();
    }
}

global_rand_gen.seed(rand_seed);
```
## 调用关系

 
![123](codestru.png) 


## Experiment Design

怎么评估泛化能力

1. 


训练集大小


 
