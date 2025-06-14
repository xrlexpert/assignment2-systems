import timeit
import argparse
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Callable, Dict, Any, Optional
from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": {
        "params": {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "rope_theta": 10000.0,
        },
        "batch_size": 4,
    },
    "medium":{
        "params": {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
            "rope_theta": 10000.0,
        },
        "batch_size": 4,
    },
    "large": {
        "params": {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 32,
            "num_heads": 20,
            "rope_theta": 10000.0,
        },
        "batch_size": 4,
    },
    "xl":{
        "params": {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 1600,
            "d_ff": 6400,
            "num_layers": 48,
            "num_heads": 25,
            "rope_theta": 10000.0,
        },
        "batch_size": 4,
    },
    "2.7B":{
        "params": {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
            "rope_theta": 10000.0,
        },
        "batch_size": 4,
    }
}


def benchmark(model_size:str, 
              mode:str, 
              warm_up_steps:int=3, 
              measure_steps:int=10,
              torch_compile:bool=False,
              use_bf16:bool=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking {model_size} on {device}")

    # Get model configuration
    model_config = MODEL_CONFIGS[model_size]
    params = model_config["params"]
    batch_size = model_config["batch_size"]

    model = BasicsTransformerLM(**params).to(device)
    if torch_compile:
        print("Using torch.compile")
        model = torch.compile(model)

    # Create input data
    print(f"Creating random data with batch size {batch_size} and context length {params['context_length']}")
    input_ids = torch.randint(0, params["vocab_size"], (batch_size, params["context_length"]), device=device)
    target_ids = torch.randint(0, params["vocab_size"], (batch_size, params["context_length"]), device=device)

    # Setup AMP context
    if use_bf16:
        print("Using bf16 mixed precision")
        amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Running {warm_up_steps} warmup steps")
    for _ in range(warm_up_steps):
        if mode == "forward":
            with amp_context:
                output = model(input_ids)
        elif mode == "forward_backward":
            with amp_context:
                output = model(input_ids)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss.backward()
        elif mode == "full_steps":
            optimizer.zero_grad()
            with amp_context:
                output = model(input_ids)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    forward_times = []
    backward_times = []
    step_times = []

    print(f"Running measurement steps{measure_steps}")
    for _ in range(measure_steps):
        if mode == "forward":
            start_time = timeit.default_timer()
            with amp_context:
                output = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - start_time)
        elif mode == "forward_backward":
            start_time = timeit.default_timer()
            with amp_context:
                output = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end_time = timeit.default_timer()
            forward_times.append(forward_end_time - start_time)

            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end_time = timeit.default_timer()
            backward_times.append(backward_end_time - forward_end_time)

        elif mode == "full_steps":
            optimizer.zero_grad()
            start_time = timeit.default_timer()
            with amp_context:
                output = model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end_time = timeit.default_timer()
            forward_times.append(forward_end_time - start_time)

            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end_time = timeit.default_timer()
            backward_times.append(backward_end_time - forward_end_time)

            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_end_time = timeit.default_timer()
            step_times.append(step_end_time - backward_end_time)

    print("--------------------------------")
    print(f"torch.compile: {torch_compile}, bf16: {use_bf16}")
    print(f"Model size: {model_size}, mode: {mode}")
    print(f"Batch size: {batch_size}, Context length: {params['context_length']}")
    print(f"Vocab size: {params['vocab_size']}")
    print(f"D_model: {params['d_model']}, D_ff: {params['d_ff']}")
    print(f"Num layers: {params['num_layers']}, Num heads: {params['num_heads']}")
    print(f"Rope theta: {params['rope_theta']}")
    print(f"Number of warmup steps: {warm_up_steps}, Number of measurement steps: {measure_steps}")

    avg_forward_time = 0.0
    avg_backward_time = 0.0
    avg_step_time = 0.0
    avg_total_time = 0.0
    if forward_times:
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_total_time += avg_forward_time
        print(f"Average forward time: {avg_forward_time:.3f} ms")
    if backward_times:
        avg_backward_time = sum(backward_times) / len(backward_times)
        avg_total_time += avg_backward_time
        print(f"Average backward time: {avg_backward_time:.3f} ms")
    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        avg_total_time += avg_step_time
        print(f"Average step time: {avg_step_time:.3f} ms")

    print(f"Average total time: {avg_total_time:.3f} ms")
    tokens_per_second = (batch_size * params["context_length"]) / avg_total_time
    print(f"Throughput: {tokens_per_second:.3f} tokens/s")

    return avg_forward_time, avg_backward_time, avg_step_time, avg_total_time


def main():
    parser = argparse.ArgumentParser(description='Run model benchmarks')
    parser.add_argument('--model_size', type=str, default="small",
                      help='Model size to benchmark (e.g., small, medium, large, xl, 2.7B)')
    parser.add_argument('--mode', type=str, default='forward_backward',
                      choices=['forward', 'forward_backward', 'full_steps'],
                      help='Benchmark mode')
    parser.add_argument('--warm_up_steps', type=int, default=3,
                      help='Number of warmup steps')
    parser.add_argument('--measure_steps', type=int, default=10,
                      help='Number of measurement steps')
    parser.add_argument('--torch_compile', action='store_true',
                      help='Use torch.compile')
    parser.add_argument('--bf16', action='store_true',
                      help='Use bf16 mixed precision')

    
    args = parser.parse_args()
    benchmark(args.model_size, args.mode, args.warm_up_steps, args.measure_steps, 
             args.torch_compile, args.bf16)

if __name__ == '__main__':
    main()
