import torch, triton
import triton.language as tl

from cs336_systems.data_whale.profile_tool import benchmark


@triton.jit
def softmax_kernel(x_ptr, o_ptr, n_col, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n_col
    x = tl.load(x_ptr + row * n_col + offs, mask = m, other=-float('inf'))
    e = tl.exp(x-tl.max(x, 0))
    y = e / tl.sum(e, 0)
    tl.store(o_ptr + row * n_col + offs, y, mask = m)


def softmax_triton(x: torch.Tensor):
    x = x.contiguous()
    o = torch.empty_like(x)
    n_col = x.shape[-1]
    n_row = x.shape[-2]
    
    grid = (n_row,)
    block_size = triton.next_power_of_2(n_col)

    softmax_kernel[grid](x, o, n_col, BLOCK=block_size)
    return o

def test():
    x = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    baseline = torch.softmax(x, dim=-1)
    test_result = softmax_triton(x)
    assert torch.allclose(test_result, baseline)
    
    def run_baseline():
        torch.softmax(x, dim=-1)
    def run():
        softmax_triton(x)

    benchmark('baseline', run_baseline)
    benchmark('triton', run)
    # Benchmark[baseline]: mean: 0.015 ms, std: 0.004 ms
    # Benchmark[triton]: mean: 0.025 ms, std: 0.005 ms
        
if __name__ == '__main__':
    test()