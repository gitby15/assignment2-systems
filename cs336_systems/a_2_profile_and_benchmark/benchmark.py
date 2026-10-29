import argparse
import statistics
from collections.abc import Callable
from timeit import default_timer

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from torch import nn

from cs336_systems.a_2_profile_and_benchmark.model_static_analysis import analyze_model_parameters

DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTENT_LEN = 512
DEFAULT_VOCAB_SIZE = 10000
DEFAULT_MODEL_SIZE = "small"


"""
按照default的content_len=512, vocab_size=10000, 模型尺寸如下：
small: 参数量122.7M, 参数内存490.7MiB
medium: 参数量403.6M, 参数内存1614.3MiB
large: 参数量924.5M, 参数内存3698.0 MiB (3GB)
xl: 参数量3249.0M, 参数内存12995.9 MiB (12GB)
10B: 参数量12238.3M, 参数内存48953.3 MiB (48GB)
"""
MODEL_CONFIGS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),  # 在L4显卡下，就不用跑这个模型了，显存装不下
}
type BatchType = tuple[torch.Tensor, torch.Tensor]

def create_model(
    size: str = DEFAULT_MODEL_SIZE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    context_length: int = DEFAULT_CONTENT_LEN,
    device: torch.device = torch.device("cpu"),
) -> BasicsTransformerLM:
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[size]

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    model_detail = analyze_model_parameters(model)
    print(f"Model created: {size} | 参数量: {model_detail['total_params'] / 1024**2:.1f}M | 参数内存: {model_detail['parameter_memory_mib']:.1f} MiB")

    return model.to(device)

@nvtx.range("#benchmark")
def benchmark(
    step: Callable[[], None],
    tag: str = '',
    warmup_steps: int = 5,
    measure_steps: int = 10,
) -> tuple[float, float]:
    if warmup_steps < 0 or measure_steps <= 0:
        raise ValueError("Invalid warmup or measurement count")


    
        # Warm-up
    for _ in range(warmup_steps):
        with nvtx.range(f"#warmup_{tag}"):
            step()

    torch.cuda.synchronize()

    # Measurement
    durations = []

    
    for _ in range(measure_steps):
        with torch.cuda.nvtx.range(f"#meansure_{tag}"):
            torch.cuda.synchronize()
            start = default_timer()
            step()
            torch.cuda.synchronize()
            end = default_timer()

        durations.append((end - start) * 1000)

    mean = statistics.mean(durations)
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0

    return mean, std


def create_batch(
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_length: int = DEFAULT_CONTENT_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    device: torch.device = torch.device("cpu"),
) -> BatchType:
    inputs = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )
    targets = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )
    return inputs, targets


def create_forward_step(model: nn.Module, batch: BatchType, device: torch.device):
    inputs, _ = batch
    model.eval()
    def step():
        with torch.inference_mode():
            model(inputs)
    return step

def create_backward_step(model: nn.Module, batch: BatchType, device: torch.device, ):
    inputs, targets = batch
    model.train()
    def step():
        loss = model(inputs, targets)
        loss.backward()
    return step


def main():
    parser = argparse.ArgumentParser(description="Benchmark a Transformer forward or training step on CUDA.")

    parser.add_argument(
        "--size",
        choices=list(MODEL_CONFIGS),
        default="small",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full"],
        default="forward",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA BF16 autocast (model parameters remain FP32).",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Run one step and print CUDA memory statistics.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    if args.batch_size <= 0 or args.context_length <= 0 or args.vocab_size <= 0:
        parser.error("batch size, context length, and vocabulary size must be positive")

    device = torch.device("cuda")
    model = create_model(
        size=args.size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device,
    )
    inputs, targets = create_batch(
        args.batch_size,
        args.context_length,
        args.vocab_size,
        device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        if args.mode == "forward":
            with torch.cuda.nvtx.range("forward"):
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.bf16,
                    ):
                        model(inputs)
            return

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.nvtx.range("forward"):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.bf16,
            ):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                )

        with torch.cuda.nvtx.range("backward"):
            loss.backward()

        if args.mode == "full":
            with torch.cuda.nvtx.range("optimizer"):
                optimizer.step()

    if args.memory_profile:
        torch.cuda.reset_peak_memory_stats(device)
        step()
        torch.cuda.synchronize(device)

        mib = 1024**2
        allocated = torch.cuda.memory_allocated(device) / mib
        reserved = torch.cuda.memory_reserved(device) / mib
        peak_allocated = torch.cuda.max_memory_allocated(device) / mib
        peak_reserved = torch.cuda.max_memory_reserved(device) / mib
#        print"")
        print(f"Allocated:      {allocated:.2f} MiB")
        print(f"Reserved:       {reserved:.2f} MiB")
        print(f"Peak allocated: {peak_allocated:.2f} MiB")
        print(f"Peak reserved:  {peak_reserved:.2f} MiB")
        print(f"|{args.mode}|{args.size}|{'bf16' if args.bf16 else 'no-bf16'}|{peak_allocated:.2f}|{peak_reserved:.2f}|")
        # print(torch.cuda.memory_summary(device=device, abbreviated=True))#        print"")
        # uv run cs336_systems/a_2_profile_and_benchmark/benchmark.py --bf16 --memory-profile --mode forward --size medium
        return

    mean, std = benchmark(
        step,
        warmup_steps=args.warmup,
        measure_steps=args.steps,
    )

    print(f"Model:  {args.size}")
    print(f"Mode:   {args.mode}")
    print(f"isBF16:  {args.bf16}")
    print(f"Batch:  {args.batch_size} x {args.context_length}")
    print(f"Mean:   {mean:.3f} ms")
    print(f"Std:    {std:.3f} ms")
    cal_type = "bf16" if args.bf16 else "no-bf16"
    print(f"|{args.mode}|{args.size}|{cal_type}|{mean:.3f}|{std:.3f}|")


if __name__ == "__main__":
    create_model(size="10B")
