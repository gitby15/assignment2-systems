import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from cs336_systems.data_whale.profile_tool import benchmark


@triton.jit
def rms_norm_kernel(x_ptr, w_ptr, o_ptr, d, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # d 表示真正的数据有多少列，offs表示在GPU中目前在哪个thread
    offs = tl.arange(0, BLOCK)
    mask = offs < d

    x = tl.load(x_ptr + row * d + offs, mask=mask, other=0)
    w = tl.load(w_ptr + offs, mask=mask, other=0)
    mean_square = tl.sum(x * x) / d
    inv_rms = tl.rsqrt(mean_square + eps)
    y = (x * inv_rms) * w
    tl.store(o_ptr + row * d + offs, y, mask=mask)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float):
    x = x.contiguous()
    w = w.contiguous()
    o = torch.empty_like(x)
    x_numel = x.numel()
    if x_numel == 0:
        return o

    d = x.shape[-1]
    n_rows = x_numel // d

    block_size = triton.next_power_of_2(d)
    rms_norm_kernel[(n_rows,)](x, w, o, d, eps, BLOCK=block_size)
    return o


def test():
    d = 768
    eps = 1e-5
    x = torch.randn(2, 7, d, device="cuda", dtype=torch.float32)
    weight = torch.empty(d, device="cuda", dtype=x.dtype).uniform_(0.5, 1.5)

    expected = F.rms_norm(x, (d,), weight=weight, eps=eps)
    actual = rms_norm(x, weight, eps)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def run_baseline():
        F.rms_norm(x, (d,), weight=weight, eps=eps)
    def run_target():
        rms_norm(x, weight, eps)
    
    benchmark('baseline', run_baseline)
    benchmark('target', run_target)


if __name__ == "__main__":
    test()
