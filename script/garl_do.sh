#22 23 has remove set sed
seeds=(123 456 789 123 456 789 123 456 789 123 456 789)
for i in 2
do
    seed=${seeds[i]}
    runid='GADO-DQN-v6-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=1 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 100 --log-interval 100 -nmb 8 --use-evo 2 -es 16 --num-envs 64 -ts 256 --ini-levels 30 --train-iter 16 --mu-op 1 --thres-hold 0.0 --dropout 0.1 --version 6 > ./logs/${runid}.log
    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy.py --train-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/train_log.log

    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy.py --test-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/test_log.log

done
