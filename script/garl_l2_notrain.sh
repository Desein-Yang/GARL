for i in {1,2,3,4}
do
    seed=456
    #seed=${seeds[i]}
    #runid='L2-SED-'${i}'-'${seed}
    runid='L2-NORMAL-'${i}'-'${seed}
    echo 'Run Exp'${runid}
    #CUDA_VISIBLE_DEVICES=0 python -m garl.train_agent --run-id ${runid} --num-steps 256 --arch nature --save-interval 100 --log-interval 100 -nmb 8 --use-evo 2 -es 16 --num-envs 32 -ts 256 --ini-levels 40 --train-iter 16 --mu-op 1 --thres-hold 0.0 --l2-weight 0.0001 --dropout 0.0 --load-seed 1 --version 6 > ./logs/${runid}.log

    #TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=1 python enjoy.py --train-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/train_best_log.log
    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICE=0 python enjoy.py --test-eval --restore-id ${runid} --arch nature --use-evo 2 --num-eval 20 --rep 50 --use-lstm 0 > ./logs/${runid}/test_best_log.log

done
