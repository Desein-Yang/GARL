seeds=(123 456 789)
for i in 30
do
    seed=${seeds[i-29]}
    runid='GA-DQN-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=1 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 2 --num-envs 20 -ts 2 -es 1 --ini-levels 10 --num-levels 20 --train-iter 4 --mu-op 1 -version 6 > ./logs/${runid}.log
done
