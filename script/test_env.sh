for i in 1
do
    runid='GA-DQN-'${i}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=0 python -m garl.test_env --run-id ${runid} --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 --use-evo 1 --ini-levels 10 --train-iter 8 -mu-op 1 > ./logs/${runid}.log
done
