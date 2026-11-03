import torch, triton
import triton.language as tl

from cs336_systems.data_whale.profile_tool import benchmark


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    mask = offs < n
    x = tl.load(x_ptr + offs, mask = mask)
    y = tl.load(y_ptr + offs, mask = mask)
    tl.store(o_ptr + offs, x + y, mask = mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int):
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    o = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return o
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, o, n, BLOCK=block_size)
    return o


def test():

    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    baseline = torch.add(x, y)
    test_result = add(x, y, 256)
    assert torch.allclose(test_result, baseline)
    
    def run_baseline():
        torch.add(x, y)
    benchmark('baseline', run_baseline)
    l = [8, 16, 32, 64, 128, 256, 512]
    for block in l:
        def run():
            add(x, y, block)
        benchmark(f'triton[{block}]', run)


if __name__ == '__main__':
    test()
        