
id='UR'
seeds=(123 456 789)
for i in 1
do
    seed=${seeds[i-1]}
    echo 'Start test '${id}-IMPALA-${i}-${seed}' size 1000'
    #CUDA_VISIBLE_DEVICES=1 python ./test_agent.py --test-eval --restore-id UR-IMPALA-${i}-${seed} --arch impala --num-eval 20 --rep 50 > ./logs/UR-IMPALA-${i}-${seed}/test1.log
    CUDA_VISIBLE_DEVICES=1 python ./enjoy.py --test-eval --restore-id UR-IMPALA-${i}-${seed} --arch impala --num-eval 20 --rep 25 > ./logs/UR-IMPALA-${i}-${seed}/test1.log
    CUDA_VISIBLE_DEVICES=1 python ./enjoy.py --test-eval --restore-id UR-IMPALA-${i}-${seed} --arch impala --num-eval 20 --rep 25 > ./logs/UR-IMPALA-${i}-${seed}/test1.log
    CUDA_VISIBLE_DEVICES=1 python ./enjoy.py --test-eval --restore-id UR-IMPALA-${i}-${seed} --arch impala --num-eval 20 --rep 25 > ./logs/UR-IMPALA-${i}-${seed}/test1.log
done
