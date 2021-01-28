
id='L2'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${id}-DQN-${i}-${seed}' size 1000'
    python enjoy_2.py --test-eval --restore-id ${id}-DQN-${i}-${seed} --num-eval 20 --rep 10 > ./logs/${id}-DQN-${i}-${seed}/test_enjoy_all.log
done
