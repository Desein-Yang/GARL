for i in 8
do
    runid='GA-DQN-'${i}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=1 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 1 --num-envs 64 --ini-levels 120 --train-iter 4 --mu-op 1 > ./logs/${runid}.log
done
