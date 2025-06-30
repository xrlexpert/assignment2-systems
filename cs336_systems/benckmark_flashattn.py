import torch
import triton.testing
import timeit
import itertools
import pandas as pd
from flash_attention_triton import FlashAttentionTriton
from flash_attention_pytorch import FlashAttentionPytorch
from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention(model:str,
                        context_length:int, 
                        d_model:int,
                        dtype:torch.dtype=torch.float32,
                        measure_steps:int=10,
                        warmup_steps:int=3,
                        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking attention with model={model}, context_length={context_length}, d_model={d_model}, dtype={dtype} on {device}")
    status = "success"
    try:
        # Make inputs require gradients for backward pass
        BATCH_SIZE = 1
        NUM_HEADS = 1
        Q = torch.randn(BATCH_SIZE, NUM_HEADS, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        K = torch.randn(BATCH_SIZE, NUM_HEADS, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        V = torch.randn(BATCH_SIZE, NUM_HEADS, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        if model == "naive":
            def attention_fn(q, k, v):
                return scaled_dot_product_attention(q, k, v)  # naive 实现不支持 is_causal
        elif model == "triton":
            def attention_fn(q, k, v):
                return FlashAttentionTriton.apply(q, k, v, True)
        elif model == "pytorch":
            def attention_fn(q, k, v):
                return FlashAttentionPytorch.apply(q, k, v, True)
        else:
            raise ValueError(f"Invalid model: {model}")
        # Warmup
        print(f"Warming up {warmup_steps} steps")
        for _ in range(warmup_steps):
            output = attention_fn(Q, K, V)
            output.backward(torch.ones_like(output, device=device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        torch.cuda.empty_cache()
        
        forward_times = []
        outputs = []
        backward_times = []
        
        # Forward pass timing
        print(f"Measuring forward pass {measure_steps} times")
        for _ in range(measure_steps):
            start_time = timeit.default_timer()
            
            output = attention_fn(Q, K, V)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - start_time)
            outputs.append(output)
        
        # Measure memory before backward
        memory_before_backward_bytes = torch.cuda.memory_allocated(device=device) if torch.cuda.is_available() else 0
        memory_before_backward_gb = memory_before_backward_bytes / (1024 ** 2)
        # Backward pass timing
        print(f"Measuring backward pass {measure_steps} times")
        for output in outputs:
            start_time = timeit.default_timer()
            output.backward(torch.ones_like(output, device=device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_times.append(timeit.default_timer() - start_time)
        
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_backward_time = sum(backward_times) / len(backward_times)
        # Clean up variables to free memory
        del Q, K, V, outputs
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        # Clean up variables to free memory
        status = "OOM"
        for var_name in ['Q', 'K', 'V', 'outputs']:
             if var_name in locals():
                 try:
                     del locals()[var_name]
                 except NameError:
                     pass
        torch.cuda.empty_cache()

    
    return {
        'model': model,
        'context_length': context_length,
        'd_model': d_model,
        'dtype': dtype,
        'avg_forward_time(ms)': round(avg_forward_time*1000, 3) if status == "success" else "-",
        'avg_backward_time(ms)': round(avg_backward_time*1000, 3) if status == "success" else "-",
        'avg_full_time(ms)': round((avg_forward_time + avg_backward_time)*1000, 3) if status == "success" else "-",
        'memory_before_backward_gb(MB)': round(memory_before_backward_gb, 3) if status == "success" else "-",
    }
    
def main():
    results = []
    context_lenghths = list(2 ** torch.arange(7, 17))
    d_models = list(2 ** torch.arange(4, 7))
    dtypes = [torch.float32, torch.bfloat16]
    for dtype in dtypes:
        for model in ["naive","triton"]:
            for context_length in context_lenghths:
                for d_model in d_models:
                    results.append(benchmark_attention(model, context_length, d_model, dtype))
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("cs336_systems/results/flash_benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()

