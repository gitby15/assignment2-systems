

import statistics
import time
import torch.nn.functional as F

from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity
# 进行基准测试（测量运行时间）和性能分析（探查函数内部的时间消耗分布）的伪代码
from cs336_systems.data_whale.mlp import run_mlp

def benchmarking():
    dim = 256
    num_layers = 4
    batch_size = 256
    num_steps = 2

    # 基准测试
    benchmark("BASE", run_mlp(dim, num_layers, batch_size, num_steps))

    scales = [2,4,6,8]
    for scale in scales:
        # step
        benchmark(f"step {scale}x", run_mlp(dim, num_layers, batch_size, num_steps*scale))

    for scale in scales:
        # dim
        benchmark(f"dim {scale}x", run_mlp(dim*scale, num_layers, batch_size, num_steps))

    for scale in scales:
        # layer
        benchmark(f"num_layers {scale}x", run_mlp(dim, num_layers*scale, batch_size, num_steps))

    for scale in scales:
        # batch_size
        benchmark(f"batch_size {scale}x", run_mlp(dim, num_layers, batch_size*scale, num_steps))
        

def profiling():
    dim = 8196
    x = torch.randn(dim, dim, device='cuda')
    y = torch.randn(dim, dim, device='cuda')        
    def ope():
        return x @ y.T

    profile("sleep", ope)
    


def benchmark(desc: str, run: Callable, num_warmups: int = 5, num_trials: int = 5) -> tuple[float, float]:
    # 暂时只考虑cuda
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    # warmup
    for _ in range(num_warmups):
        run()

    # 清空cuda的异步计算任务
    torch.cuda.synchronize()

    times: list[float] = []

    for _ in range(num_trials):
        # torch.cuda.synchronize()
        start_time = time.time() * 1000
        run()
        torch.cuda.synchronize()
        end_time = time.time() * 1000
        times.append(end_time - start_time)

    mean = statistics.mean(times)
    std = statistics.stdev(times)
    print(f"Benchmark[{desc}]: mean: {mean:.3f} ms, std: {std:.3f} ms")
    return mean, std


def profile(desc: str, run: Callable, num_warmups: int = 5, with_stack: bool = False):
    # warmups
    for _ in range(num_warmups):
        run()
    # 清空cuda的异步计算任务
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=with_stack) as prof:
        run()
        torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total")
    #text(f"## {description}")
    #text(table, verbatim=True)
    # Write stack trace visualization
    # if with_stack:
    #     text_path = f"var/stacks_{desc}.txt"
    #     svg_path = f"var/stacks_{desc}.svg"
    #     prof.export_stacks(text_path, "self_cuda_time_total")
    print(f"======== Profile[{desc}] ========")
    print(table)
    


if __name__ == "__main__":
    pass
    