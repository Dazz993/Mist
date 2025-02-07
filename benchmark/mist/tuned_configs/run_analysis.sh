# Change the path to the mist exec folder to the correct path
MIST_HOME=/workspace

cd $MIST_HOME/benchmark/mist/analysis

run_single_node_analysis_given_a_folder() {
    folder=$1
    for file in "$folder"/*; do
        if [[ $file =~ -n_([0-9]+)-m_([0-9]+) ]]; then
        # Extract n and m
        n="${BASH_REMATCH[1]}"
        m="${BASH_REMATCH[2]}"
        
        # Skip if n is not 1
        if [[ "$n" -ne 1 ]]; then
            continue
        fi

        # Extract filename
        filename=$(basename "$file")
        echo "Folder: $folder, Filename: $filename -> n=$n, m=$m"

        set -x
        
        python run.py \
            --config-path $folder \
            --config-name $filename \
            +output_path=$file

        sleep 1

        fi

    done
}

run_single_node_analysis_given_a_folder $MIST_HOME/benchmark/mist/tuned_configs/l4-24gb/gpt
run_single_node_analysis_given_a_folder $MIST_HOME/benchmark/mist/tuned_configs/l4-24gb/llama