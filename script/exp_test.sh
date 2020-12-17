i=1
id='L2'
seed=123
CUDA_VISIBLE_DEVICES=0 python test_restore.py --test-eval --restore-id ${id}-DQN-${i}-${seed} --num-eval 20 --rep 50 > ./logs/${id}-DQN-${i}-${seed}/test.log

