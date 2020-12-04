
#mpirun -np 8 python coinrun.train_agent --run-id origin --num-levels 500 --set-seed 13 --num-steps 256 --num-envs 32 --arch nature --save-interval 10 > ./log.txt
runid='dropout2'
CUDA_VISIBLE_DEVICES=1 python -m coinrun.train_agent --run-id ${runid} --num-levels 500 --dropout 0.05 --num-envs 64 --num-steps 256 --arch nature --save-interval 500 --log-interval 100 --num-eval 1000 -nmb 4 > ./logs/${runid}.log
