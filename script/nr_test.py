
runid='NR'
seeds=(123 456 789)
for i in {1,2,3}
do
    seed=${seeds[i-1]}
    echo 'Start test '${runid}-DQN-${i}-${seed}' size 1000'
    CUDA_VISIBLE_DEVICES=0 python test_restore.py --test-eval --restore-id ${runid}-DQN-${i}-${seed} --arch nature --set-seed 483 --num-levels 0 --rep 50 > ./logs/${runid}-DQN-${i}-${seed}/test.log
done
