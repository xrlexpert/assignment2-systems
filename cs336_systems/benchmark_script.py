import timeit
import torch
import torch.nn.functional as F
from typing import Callable
import argparse
import importlib
import gc

def run_forward_step(model_class, params:dict, batch_size:int=4, num_steps:int=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def run():
        # Only moved creation inside, no explicit cleanup
        model = model_class(**params).to(device)
        input_ids = torch.randint(0, params["vocab_size"], (batch_size, params["context_length"]), device=device)
        
        for _ in range(num_steps):
            output = model(input_ids)
        del model
        del input_ids
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return run

def run_backward_step(model_class, params:dict, batch_size:int=4, num_steps:int=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def run():
        # Only moved creation inside, no explicit cleanup
        model = model_class(**params).to(device)
        input_ids = torch.randint(0, params["vocab_size"], (batch_size, params["context_length"]), device=device)
        target_ids = input_ids[:, 1:]
        
        for _ in range(num_steps):
            output = model(input_ids[:, :-1])
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.reshape(-1))
            loss.backward()
        del model
        del input_ids
        del target_ids
        del output
        del loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    return run

def benchmark(description: str, run: Callable, warm_up_steps:int=0, measure_steps:int=1):
    times = []
    print(f"Benchmarking {description}")
    
    for _ in range(warm_up_steps):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(measure_steps):
        start_time = timeit.default_timer()
        run()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(timeit.default_timer() - start_time)

    return times

def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance.')
    parser.add_argument('--model_class', type=str, default='cs336_basics.BasicTransformerLM', help='Model class to load.')
    parser.add_argument('--params', type=dict, required=True, help='Model parameters.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the model.')
    parser.add_argument('--backward', action='store_true', help='Whether to perform backward pass.')
    args = parser.parse_args()

    # Load the model class dynamically
    module, cls_name = args.model_class.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module), cls_name)
    params = args.params
    batch_size = args.batch_size
    backward = args.backward

    if backward:
        run = run_backward_step(model_class, params, batch_size)
    else:
        run = run_forward_step(model_class, params, batch_size)

    times = benchmark(f"{args.model_class} with {params}", run)
    print(f"Average time: {sum(times) / len(times):.4f} seconds")

if __name__ == "__main__":
    main()
