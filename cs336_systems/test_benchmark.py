import unittest
import pandas as pd
from cs336_systems.benchmark_script import run_forward_step, run_backward_step, benchmark
from cs336_basics.model import BasicsTransformerLM

TESTS = {
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

class TestBenchmark(unittest.TestCase):
    def test_benchmark_results(self):
        results = []
        for model_name, test_config in TESTS.items():
            params = test_config["params"]
            batch_size = test_config["batch_size"]
            forward_run = run_forward_step(BasicsTransformerLM, params, batch_size)
            backward_run = run_backward_step(BasicsTransformerLM, params, batch_size)

            forward_times = benchmark(f"{model_name} Forward Step", forward_run)
            backward_times = benchmark(f"{model_name} Backward Step", backward_run)

            results.append({
                "Model": model_name,
                "Batch Size": batch_size,
                "Context Length": params["context_length"],
                "Vocab Size": params["vocab_size"],
                "D Model": params["d_model"],
                "D FF": params["d_ff"],
                "Num Layers": params["num_layers"],
                "Num Heads": params["num_heads"],
                "Forward Average Time (ms)": sum(forward_times) / len(forward_times) * 1000,
                "Backward Average Time (ms)": sum(backward_times) / len(backward_times) * 1000
            })

        df = pd.DataFrame(results)
        print(df.to_latex(index=False))

if __name__ == "__main__":
    unittest.main() 