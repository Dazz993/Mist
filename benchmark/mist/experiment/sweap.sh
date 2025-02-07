run_experiment() {
    model=$1
    n=$2
    m=$3
    seq=$4
    device_to_use=$5

    set -x
    CUDA_VISIBLE_DEVICES=$5 python run.py --model $model -n $n -m $m -s $seq --skip-exec
}

# run_experiment falcon/7b 1 8 2048
# run_experiment falcon/7b 2 8 2048
# run_experiment falcon/7b 4 8 2048
# run_experiment falcon/7b 8 8 2048
# run_experiment falcon/7b 16 8 2048
# run_experiment falcon/7b 32 8 2048
# run_experiment falcon/7b 64 8 2048

# run_experiment falcon/40b 4 8 2048
# run_experiment falcon/40b 8 8 2048
# run_experiment falcon/40b 16 8 2048
# run_experiment falcon/40b 32 8 2048
# run_experiment falcon/40b 64 8 2048
# run_experiment falcon/40b 128 8 2048
# run_experiment falcon/40b 256 8 2048

# run_experiment llama/7b 1 8 2048
# run_experiment llama/7b 2 8 2048
# run_experiment llama/7b 4 8 2048
# run_experiment llama/7b 8 8 2048
# run_experiment llama/7b 16 8 2048
# run_experiment llama/7b 32 8 2048
# run_experiment llama/7b 64 8 2048

# run_experiment llama/13b 2 8 2048
# run_experiment llama/13b 4 8 2048
# run_experiment llama/13b 8 8 2048
# run_experiment llama/13b 16 8 2048
# run_experiment llama/13b 32 8 2048
# run_experiment llama/13b 64 8 2048
# run_experiment llama/13b 128 8 2048

# run_experiment llama/70b 16 8 4096
# run_experiment llama/70b 32 8 4096
# run_experiment llama/70b 64 8 4096
# run_experiment llama/70b 128 8 4096
# run_experiment llama/70b 256 8 4096
# run_experiment llama/70b 512 8 4096

# (
# gpu=0
# run_experiment falcon/7b 1 8 2048 $gpu
# run_experiment falcon/7b 2 8 2048 $gpu
# run_experiment falcon/7b 4 8 2048 $gpu
# run_experiment falcon/7b 8 8 2048 $gpu
# run_experiment falcon/7b 16 8 2048 $gpu
# run_experiment falcon/7b 32 8 2048 $gpu
# run_experiment falcon/7b 64 8 2048 $gpu
# ) &

# (
# gpu=1
# run_experiment falcon/40b 4 8 2048 $gpu
# run_experiment falcon/40b 8 8 2048 $gpu
# run_experiment falcon/40b 16 8 2048 $gpu
# run_experiment falcon/40b 32 8 2048 $gpu
# run_experiment falcon/40b 64 8 2048 $gpu
# ) &

# (
# gpu=2
# run_experiment falcon/40b 128 8 2048 $gpu
# run_experiment falcon/40b 256 8 2048 $gpu
# ) &


# (
# gpu=3
# run_experiment llama/7b 1 8 2048 $gpu
# run_experiment llama/7b 2 8 2048 $gpu
# run_experiment llama/7b 4 8 2048 $gpu
# run_experiment llama/7b 8 8 2048 $gpu
# run_experiment llama/7b 16 8 2048 $gpu
# run_experiment llama/7b 32 8 2048 $gpu
# run_experiment llama/7b 64 8 2048 $gpu
# ) &

# (
# gpu=4
# run_experiment llama/13b 2 8 2048 $gpu
# run_experiment llama/13b 4 8 2048 $gpu
# run_experiment llama/13b 8 8 2048 $gpu
# run_experiment llama/13b 16 8 2048 $gpu
# run_experiment llama/13b 32 8 2048 $gpu
# ) &

# (
# gpu=5
# run_experiment llama/13b 64 8 2048 $gpu
# run_experiment llama/13b 128 8 2048 $gpu
# ) &

(
gpu=6
run_experiment llama/70b 16 8 4096 $gpu
run_experiment llama/70b 32 8 4096 $gpu
run_experiment llama/70b 64 8 4096 $gpu
run_experiment llama/70b 128 8 4096 $gpu
) &

(
gpu=7
run_experiment llama/70b 256 8 4096 $gpu
run_experiment llama/70b 512 8 4096 $gpu
) &

