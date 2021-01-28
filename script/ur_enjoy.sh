
id='UR'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${id}-DQN-${i}-${seed}' size 1000'
    python enjoy_2.py --train-eval --restore-id dropout${i} --arch nature --num-eval 20 --rep 50 > ./logs/dropout${i}/train_enjoy.log
done
