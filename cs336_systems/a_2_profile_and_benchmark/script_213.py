# Problem (benchmarking_script):  Benchmarking Script (4 points)

"""
(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass,
and optimizer step in your model. Specifically, your script should support the following:
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛
steps (either only forward, forward and backward, or forward and backward with optimizer
step, depending on an argument). For timing, you can use the Python timeit module (e.g.,
either using the timeit function, or using timeit.default_timer(), which gives you the
system’s highest resolution clock, thus a better default for benchmarking than
time.time()).
• Call torch.cuda.synchronize() after each step.
"""
from torch import nn
import torch

from cs336_systems.a_2_profile_and_benchmark.benchmark import benchmark, create_batch, create_forward_step, create_model


"""
(b) Time the forward, backward, and optimizer step for the model sizes described in
Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of
timings over 10 measurement steps. How long does a forward pass take? How about a
backward pass? Do you see high variability across measurements, or is the standard deviation
small?
"""

def question_b():
    device = torch.device("cuda")
    

    model_size = [
        "small",
        "medium",
        "large",
        "xl",
        # "10B", # 我的L4显卡跑不动
    ]

    for size in model_size:
        model = create_model(size=size, device=device)
        batch = create_batch(device=device)
        forward_step = create_forward_step(model, batch, device=device)
        try:
            mean, std = benchmark(forward_step, warmup_steps=5, measure_steps=10)
            print(f"Model: {size} | Mean: {mean:.3f} ms | Std: {std:.3f} ms")
        except Exception as e:
            print(f"Error benchmarking {size}: {e}")

if __name__ == "__main__":
    question_b()


"""
输出结果
Model created: small | 参数量: 122.7M | 参数内存: 490.7 MiB
Model: small | Mean: 100.493 ms | Std: 1.900 ms
Model created: medium | 参数量: 403.6M | 参数内存: 1614.3 MiB
Model: medium | Mean: 299.096 ms | Std: 2.215 ms
Model created: large | 参数量: 924.5M | 参数内存: 3698.0 MiB
Model: large | Mean: 648.590 ms | Std: 2.400 ms
Model created: xl | 参数量: 3249.0M | 参数内存: 12995.9 MiB
Model: xl | Mean: 1942.981 ms | Std: 7.213 ms
"""