
#mpirun -np 8 python coinrun.train_agent --run-id origin --num-levels 500 --set-seed 13 --num-steps 256 --num-envs 32 --arch nature --save-interval 10 > ./log.txt
CUDA_VISIBLE_DEVICES=0 python -m coinrun.train_agent --run-id test_time_1127 --num-levels 500 --num-steps 256 --arch nature --save-interval 5 > ./logs/screen.log
