
#mpirun -np 8 python coinrun.train_agent --run-id origin --num-levels 500 --set-seed 13 --num-steps 256 --num-envs 32 --arch nature --save-interval 10 > ./log.txt
seeds=(123 456 789)
for i in {1,2}
do
    seed=${seeds[i-1]}
    runid=UR-DQN-${i}-${seed}
    echo 'Run Exp'${runid}
    CUDA_VISIBLE_DEVICES=1 python -m coinrun.train_agent --run-id ${runid} --num-levels 500 --num-envs 64 --num-steps 256 --arch nature --save-interval 500 --log-interval 100 -nmb 8 -set-seed ${seed}> ./logs/${runid}.log
done
