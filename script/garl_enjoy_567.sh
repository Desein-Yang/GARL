id='GA'
seeds=(123 456 789)
for i in 14
do
    seed=${seeds[i-14]}
    echo 'Start test '${id}-DQN-${i}' size 1000'
    RCALL_NUM_GPU=1 CUDA_VISIBLE_DEVICES=1 python enjoy_2.py --test-eval --restore-id ${id}-DQN-${i} --arch nature --use-evo 1 --num-eval 20 --rep 50 > ./logs/${id}-DQN-${i}/test_enjoy.log
done
