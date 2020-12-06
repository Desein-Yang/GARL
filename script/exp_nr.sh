

runid='netrandom0'
CUDA_VISIBLE_DEVICES=0 python -m coinrun.train_agent --run-id ${runid} --num-levels 500 --num-envs 64 --num-steps 256 --arch impala --save-interval 500 --log-interval 100 --set-seed 456 -nmb 8 --use-lstm 2 --fm-coeff 0.002 > ./logs/${runid}.log

