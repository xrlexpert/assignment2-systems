#!/bin/bash

# Model sizes
MODEL_SIZES="small medium large xl 2.7B"

# Context lengths
CONTEXT_LENGTHS="128 256 512 1024"

# Create results directory inside cs336_systems
mkdir -p cs336_systems/profiling_results

# Loop through each model size and context length
for model_size in $MODEL_SIZES; do
    for ctx_len in $CONTEXT_LENGTHS; do
        echo "Running profiling for model: $model_size, context length: $ctx_len"
        
        # Run profiling with nsys
        uv run nsys profile -o "cs336_systems/profiling_results/${model_size}_${ctx_len}_result" \
            python ./cs336_systems/benchmark_script.py \
            --model_size "$model_size" \
            --mode full_steps \
            --context_length "$ctx_len" \
            --warm_up_steps 3 \
            --measure_steps 10
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Successfully completed profiling for $model_size with context length $ctx_len"
        else
            echo "Failed to complete profiling for $model_size with context length $ctx_len"
        fi
        
        # Add a small delay between runs
        sleep 2
    done
done

echo "All profiling runs completed!" 