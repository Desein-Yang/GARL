seeds=(123 456 789)
for i in 13
do
    seed=${seeds[i-11]}
    runid='GA-DQN-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=0 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 1 --num-envs 32 -ts 16 -es 1 --ini-levels 10 --train-iter 8 --mu-op 1 > ./logs/${runid}.log
done
