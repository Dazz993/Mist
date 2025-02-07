run_experiment() {
    model=$1
    n=$2
    m=$3
    seq=$4
    device_to_use=$5
    global_batch_size=$6

    set -x
    CUDA_VISIBLE_DEVICES=$5 python run.py --model $model -n $n -m $m -s $seq --global-batch-size $global_batch_size --skip-exec
}

run_experiment falcon/40b  64 8 4096 0 8192
