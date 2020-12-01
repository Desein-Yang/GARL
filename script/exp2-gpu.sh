
#mpirun -np 8 python coinrun.train_agent --run-id origin --num-levels 500 --set-seed 13 --num-steps 256 --num-envs 32 --arch nature --save-interval 10 > ./log.txt
runid='testNov30'
CUDA_VISIBLE_DEVICES=0 python -m coinrun.train_agent --run-id ${runid} --num-levels 500 --num-envs 16 --num-steps 256 --arch nature --save-interval 100 > ./logs/${runid}.log
