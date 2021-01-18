id='GA'
seeds=(123 456 789)
for i in 5
do
    seed=${seeds[i-5]}
    echo 'Start test '${id}-DQN-${i}' size 1000'
    RCALL_NUM_GPU=1 CUDA_VISIBLE_DEVICES=1 python enjoy.py --test-eval --restore-id ${id}-DQN-${i} --arch nature --use-evo 1 --run-id ${id}-DQN-${i} --num-eval 20 --num-levels 1000 --rep 50 > ./logs/${id}-DQN-${i}/test_enjoy.log
done
