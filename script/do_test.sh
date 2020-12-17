
id='DO'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${id}-DQN-${i}-${seed}' size 1000'
    CUDA_VISIBLE_DEVICES=0 python test_restore.py --test-eval --restore-id ${id}-DQN-${i}-${seed} --arch nature --num-eval 20 --rep 50 > ./logs/${id}-DQN-${i}-${seed}/test.log
done
