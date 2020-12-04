
#mpirun -np 8 python coinrun.train_agent --run-id origin --num-levels 500 --set-seed 13 --num-steps 256 --num-envs 32 --arch nature --save-interval 10 > ./log.txt
runid='l2rg-2'
seed=456
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=1 python -m coinrun.train_agent --run-id ${runid} --num-levels 500 --l2 0.5 --num-envs 32 --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 -set-seed ${seed}> ./logs/${runid}.log
