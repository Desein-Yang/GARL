#22 23 has remove set sed
seeds=(123 456 789 123 456 789 123 456 789 123 456 789)
for i in 1
do
    seed=${seeds[i]}
    runid='GADO-DQN-v6-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=1 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 100 --log-interval 100 -nmb 8 --use-evo 2 -es 16 --num-envs 64 -ts 256 --ini-levels 30 --train-iter 16 --mu-op 1 --thres-hold 0.0 --dropout 0.1 --version 6 > ./logs/${runid}.log
done
