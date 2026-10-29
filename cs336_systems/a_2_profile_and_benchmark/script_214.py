# Problem (nsys_profile):  Nsight Systems Profiling (5 points)


"""
Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes
from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where
the largest available size should be the longest context length you can fit in memory. Pick the
combinations you think would be the most interesting to look at. For each profile answer the
following questions:
"""
import torch
from cs336_basics.model import annotated_scaled_dot_product_attention
import cs336_basics.model

from cs336_systems.a_2_profile_and_benchmark.benchmark import benchmark
from cs336_systems.a_2_profile_and_benchmark.benchmark import create_batch, create_forward_step, create_model


def profil_nsys():
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    device = torch.device('cuda')

    # 对于forward，的input，做两组实验：
    # 1. batch相同，长度不同，观察性能表现
    # 2. 长度相同，不同的batch，观察性能表现

    default_batch = 5
    default_content = 512
    diff_content = [128, 512, 1024, 2048, 4096] # 8192就OOM了，需要申请15GB
    diff_batch = [4,  16, 64, 256]
    model_size = 'small'
    for content_length in diff_content:
        model = create_model(size=model_size, context_length=content_length, device=device)
        batch = create_batch(batch_size=default_batch, context_length=content_length, device=device)
        forward_step = create_forward_step(model, batch, device=device)
        mean, std = benchmark(forward_step, tag=f"{model_size}_{default_batch}_{content_length}", measure_steps=5)
        print(f"[Model]: {model_size} | Batch Size: {default_batch} + {content_length} | Mean: {mean:.3f} ms | Std: {std:.3f} ms")
        del model
        del batch
        del forward_step
        
    # for batch_size in diff_batch:
    #     model = create_model(size=model_size, context_length=default_content, device=device)
    #     batch = create_batch(batch_size=batch_size, context_length=default_content, device=device)
    #     forward_step = create_forward_step(model, batch, device=device)
    #     mean, std = benchmark(forward_step, tag=f"{model_size}_{batch_size}_{default_content}", measure_steps=5)
    #     print(f"[Model]: {model_size} | Batch Size: {batch_size} + {default_content} | Mean: {mean:.3f} ms | Std: {std:.3f} ms")
    #     del model
    #     del batch
    #     del forward_step
        


if __name__ == "__main__":
    # 生成报告
    # uv run nsys profile -o nsys-report/small-forward -- python cs336_systems/a_2_profile_and_benchmark/script_214.py

    # 快速查看报告
    # nsys stats --report nvtx_gpu_proj_trace nsys-report/small-forward.nsys-rep >> ./nsys-report/small-forward.txt
    profil_nsys()


"""
Model created: small | 参数量: 122.7M | 参数内存: 490.7 MiB
[Model]: small | Batch Size: 5 + 128 | Mean: 23.997 ms | Std: 0.215 ms
[Model]: small | Batch Size: 5 + 512 | Mean: 124.823 ms | Std: 1.511 ms
[Model]: small | Batch Size: 5 + 1024 | Mean: 336.683 ms | Std: 1.973 ms
[Model]: small | Batch Size: 5 + 2048 | Mean: 1027.050 ms | Std: 0.805 ms
[Model]: small | Batch Size: 5 + 4096 | Mean: 3593.706 ms | Std: 3.226 ms
"""




