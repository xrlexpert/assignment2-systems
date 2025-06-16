import torch
import timeit
from cs336_basics.model import scaled_dot_product_attention
import pandas as pd
def benchmark_attention(context_length:int, 
                        batch_size:int,
                        d_model:int,
                        measure_steps:int=10,
                        warmup_steps:int=3,
                        ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking attention with context_length={context_length}, batch_size={batch_size}, d_model={d_model}, measure_steps={measure_steps}, warmup_steps={warmup_steps} on {device}")
    status = "success"
    try:
        # Make inputs require gradients for backward pass
        Q = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
        K = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
        V = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)

        # Warmup
        print(f"Warming up {warmup_steps} steps")
        for _ in range(warmup_steps):
            output = scaled_dot_product_attention(Q, K, V)
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
            
            output = scaled_dot_product_attention(Q, K, V)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - start_time)
            outputs.append(output)
        
        # Measure memory before backward
        memory_before_backward_bytes = torch.cuda.memory_allocated(device=device) if torch.cuda.is_available() else 0
        memory_before_backward_gb = memory_before_backward_bytes / (1024 ** 3)
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
        'context_length': context_length,
        'd_model': d_model,
        'avg_forward_time(s)': round(avg_forward_time, 3) if status == "success" else "-",
        'avg_backward_time(s)': round(avg_backward_time, 3) if status == "success" else "-",
        'memory_before_backward_gb(GB)': round(memory_before_backward_gb, 3) if status == "success" else "-"
    }
    
if __name__ == "__main__":

    batch_size = 8
    context_lengths = [256, 1024, 4096, 8192, 16384]
    d_models = [16, 32, 64, 128]
    results = []
    for context_length in context_lengths:
        for d_model in d_models:
            results.append(benchmark_attention(context_length, batch_size, d_model))
    
    results_latex = pd.DataFrame(results).to_latex(index=False)
    print(f"--------------------------------")
    print(results_latex)
