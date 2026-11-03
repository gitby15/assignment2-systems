import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_gate_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    gate = tl.load(gate_ptr + offs, mask=mask, other=0)
    up = tl.load(up_ptr + offs, mask=mask, other=0)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    out = (gate * sigmoid) * up

    tl.store(out_ptr + offs, out, mask=mask)


def swiglu_gate_triton(gate: torch.Tensor, up: torch.Tensor):
    assert gate.shape == up.shape

    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)

    n = gate.numel()
    if n == 0:
        return out

    block = 256
    grid = (triton.cdiv(n, block),)
    swiglu_gate_kernel[grid](gate, up, out, n, BLOCK=block)
    return out