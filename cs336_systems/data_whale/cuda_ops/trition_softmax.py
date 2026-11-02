


import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)          # 每个块负责一行
    row_start = pid * n_cols        # 该行的起始地址
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * n_cols  # 确保不越界

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)       # 求最大值（数值稳定）
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum
    tl.store(y_ptr + offsets, y, mask=mask)


def row_sum_kenel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # pid是行号
    row_start = pid * N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        offsets = row_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += x   # 累加当前tile

    result = tl.sum(acc, axis=0)     # 跨线程归约求和
    tl.store(y_ptr + pid, result)    # 标量结果写回
