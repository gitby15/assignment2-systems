
import argparse
import statistics
from collections.abc import Callable
from timeit import default_timer

import torch

from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}


def benchmark(
    step: Callable[[], None],
    warmup_steps: int,
    measure_steps: int,
) -> tuple[float, float]:
    if warmup_steps < 0 or measure_steps <= 0:
        raise ValueError("Invalid warmup or measurement count")

    # Warm-up
    for _ in range(warmup_steps):
        step()

    torch.cuda.synchronize()

    # Measurement
    durations = []

    for _ in range(measure_steps):
        with torch.cuda.nvtx.range('measured_step'):
            start = default_timer()
            step()
            torch.cuda.synchronize()
            end = default_timer()

        durations.append((end - start) * 1000)

    mean = statistics.mean(durations)
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0

    return mean, std


def create_model(
    size: str,
    context_length: int,
    vocab_size: int,
    device: torch.device,
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

    return model.to(device)


def create_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a Transformer forward or training step on CUDA."
    )

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
        args.size,
        args.context_length,
        args.vocab_size,
        device,
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

        print("====")
        print(f"Allocated:      {allocated:.2f} MiB")
        print(f"Reserved:       {reserved:.2f} MiB")
        print(f"Peak allocated: {peak_allocated:.2f} MiB")
        print(f"Peak reserved:  {peak_reserved:.2f} MiB")
        print(f"|{args.mode}|{args.size}|{'bf16' if args.bf16 else 'no-bf16'}|{peak_allocated:.2f}|{peak_reserved:.2f}|")
        # print(torch.cuda.memory_summary(device=device, abbreviated=True))
        print("====")
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
    main()
