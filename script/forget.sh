#22 23 has remove set sed
seeds=(123 456 789 123 456 789 123 456 789)
for i in 1
do
    seed=${seeds[i]}
    runid='GA-Forget-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=0 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 2 -es 1 --num-envs 64 -ts 32 --ini-levels 20 --train-iter 8 --mu-op 1 > ./logs/${runid}.log
done
