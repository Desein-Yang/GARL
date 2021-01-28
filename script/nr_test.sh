
runid='NR'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${runid}-DQN-${i}-${seed}' size 1000'
    python enjoy_2.py --test-eval --restore-id ${runid}-DQN-${i}-${seed} --use-lstm 2 --arch nature --num-eval 20 --rep 50 > ./logs/${runid}-DQN-${i}-${seed}/curve_enjoy.log
    #python enjoy_2.py --train-eval --restore-id ${runid}-DQN-${i}-${seed} --use-lstm 2 --arch nature --num-eval 20 --rep 50 > ./logs/${runid}-DQN-${i}-${seed}/train_enjoy.log
done
