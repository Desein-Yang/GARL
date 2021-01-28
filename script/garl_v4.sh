#22 23 has remove set sed
seeds=(123 456 789 123 456 789 123 456 789)
for i in 3
do
    seed=${seeds[i]}
    runid='GA-DQN-v4-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=0 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 2 -es 16 --num-envs 64 -ts 256 --ini-levels 100 --train-iter 16 --mu-op 1 --thres-hold 0.05 > ./logs/${runid}.log
done
