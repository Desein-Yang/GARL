id='GA'
seeds=(123 456 789)
for i in 3
do
    seed=123
    #seed=${seeds[i-20]}
    runid=${id}-DQN-v4-${i}-${seed}
    echo 'Start test '${runid}' size 1000'
    #TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy_2.py --train-eval --restore-id ${id}-DQN-${i}-${seed} --arch nature --use-evo 1 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${id}-DQN-${i}-${seed}/train_enjoy_lowd.log

    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy_2.py --test-eval --restore-id ${runid} --arch nature --use-evo 1 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/test_enjoy_lowd.log
done
