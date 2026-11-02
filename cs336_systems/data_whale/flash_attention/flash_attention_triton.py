import torch
import triton
import triton.language as tl

@triton.jit
def _attention_fwd(
    Q, K, V, O,
    # Q/K/V 各自的四个 stride
    sq_b: tl.constexpr, sq_h: tl.constexpr,
    sq_t: tl.constexpr, sq_d: tl.constexpr,
    sk_b: tl.constexpr, sk_h: tl.constexpr,
    sk_t: tl.constexpr, sk_d: tl.constexpr,
    sv_b: tl.constexpr, sv_h: tl.constexpr,
    sv_t: tl.constexpr, sv_d: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 每个 program 对应一个 Q 块，以及一个 batch/head
    q_block_id = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    # 当前 program 内的序列位置、特征位置
    rows = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    cols = tl.arange(0, BLOCK_K)
    dims = tl.arange(0, D)

    # 定位当前 batch/head 的起始地址
    q_base = Q + b * sq_b + h * sq_h
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h

    # 加载 Q 块：[BLOCK_Q, D]
    q_ptrs = (
        q_base
        + rows[:, None] * sq_t
        + dims[None, :] * sq_d
    )
    q = tl.load(q_ptrs, mask=rows[:, None] < T, other=0)

    # online softmax 状态，使用 FP32
    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, D), tl.float32)

    for start in range(0, T, BLOCK_K):
        keys = start + cols

        # K 直接以转置后的形状加载：[D, BLOCK_K]
        k_ptrs = (
            k_base
            + dims[:, None] * sk_d
            + keys[None, :] * sk_t
        )
        k = tl.load(k_ptrs, mask=keys[None, :] < T, other=0)

        # [BLOCK_Q, D] @ [D, BLOCK_K]
        scores = tl.dot(q, k, input_precision="tf32x3") * SCALE

        # 越界 key 必须排除出 softmax
        scores = tl.where(
            keys[None, :] < T,
            scores,
            -float("inf"),
        )

        # 与 PyTorch 版本完全相同的状态更新
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        weights = tl.exp(scores - m_new[:, None])

        l = alpha * l + tl.sum(weights, axis=1)

        # V 块：[BLOCK_K, D]
        v_ptrs = (
            v_base
            + keys[:, None] * sv_t
            + dims[None, :] * sv_d
        )
        v = tl.load(v_ptrs, mask=keys[:, None] < T, other=0)

        # 输入和累加器均使用 FP32，避免默认 TF32 的精度损失。
        acc = acc * alpha[:, None]
        acc = tl.dot(weights, v, acc, input_precision="tf32x3")
        m = m_new

    out = acc / l[:, None]

    # O 由 wrapper 分配为连续的 [B, H, T, D]
    o_ptrs = (
        O
        + bh * T * D
        + rows[:, None] * D
        + dims[None, :]
    )
    tl.store(o_ptrs, out, mask=rows[:, None] < T)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: tuple[int, int]):
    """FP32 非 causal attention，输入为 [B, H, T, D]；仅支持前向推理。"""
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q、k、v 必须具有相同的 [B, H, T, D] 形状")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("q、k、v 必须位于同一块 CUDA 设备")
    if q.dtype != torch.float32:
        raise ValueError("当前 kernel 要求 float32 输入")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q、k、v 的 dtype 必须相同")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("尚未实现 backward，请在 torch.inference_mode() 下调用")

    B, H, T, D = q.shape
    if D < 16 or D & (D - 1):
        raise ValueError("当前 kernel 要求 D 是不小于 16 的 2 的幂")

    # 必须连续：kernel 按 [B, H, T, D] 的连续布局写入 O。
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if output.numel() == 0:
        return output

    block_q, block_k = block_size
    grid = (triton.cdiv(T, block_q), B * H)
    with torch.cuda.device(q.device):
        _attention_fwd[grid](
            q, k, v, output,
            *q.stride(), *k.stride(), *v.stride(),
            H=H,
            T=T,
            D=D,
            SCALE=D ** -0.5,
            BLOCK_Q=block_q,
            BLOCK_K=block_k
        )
    return output
