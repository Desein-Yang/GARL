
id='DO'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test ${id}-DQN-${i}-${seed} size 1000'
    #RCALL_NUM_GPU=1 CUDA_VISIBLE_DEVICES=-1 python enjoy_2.py --train-eval --restore-id ${id}-DQN-${i}-${seed} --num-eval 20 --num-levels 1000 --rep 50 > ./logs/${id}-DQN-${i}-${seed}/train_enjoy.log
    RCALL_NUM_GPU=1 CUDA_VISIBLE_DEVICES=-1 python enjoy_2.py --test-eval --restore-id ${id}-DQN-${i}-${seed} --num-eval 20 --num-levels 1000 --rep 50 > ./logs/${id}-DQN-${i}-${seed}/curve_enjoy.log
done
