
id='L2'
seeds=(123 456 789)
for i in {2 3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${id}-DQN-${i}-${seed}' size 1000'
    RCALL_NUM_GPU=1 CUDA_VISIBLE_DEVICES=-1 python enjoy.py --test-eval --restore-id ${id}-DQN-${i}-${seed} --num-eval 20 --num-levels 1000 --rep 50 > ./logs/${id}-DQN-${i}-${seed}/test_enjoy.log
done
