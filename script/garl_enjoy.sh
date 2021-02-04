id='GA'
seeds=(123 456 789)
for i in 8
do
    seed=789
    #seed=${seeds[i-20]}
    runid=${id}-DQN-v6-${i}-${seed}
    echo 'Start test '${runid}' size 1000'
    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy.py --train-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/train_log.log

    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy.py --test-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/test_log.log
done
