import torch
import torch.nn.functional as F

from torch import nn

from cs336_systems.data_whale.profile_tool import benchmark, profile
from cs336_systems.data_whale.flash_attention.flash_attention_triton import flash_attention


class torchMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        T = x.size(1)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        # [B, T, C] => [B, T, H ,D] => [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        # [B, H, T, D] => [B, T, H, D] => [B, T, C]
        o = o.transpose(1, 2).contiguous().view(B, T, self.dim)
        return o


class NormalMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        T = x.size(1)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # [B, T, C] => [B, T, H ,D] => [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 这部分需要替换成Flash Attention
        p = q @ k.transpose(-2, -1)
        p = p / (self.head_dim**0.5)
        s = F.softmax(p, dim=-1)
        o = s @ v
        # 这部分需要替换成Flash Attention
        # [B, H, T, D] => [B, T, H, D] => [B, T, C]
        o = o.transpose(1, 2).contiguous().view(B, T, self.dim)
        return o


def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_q=16, block_k=16):
    original_dtype = q.dtype

    q, k, v = q.float(), k.float(), v.float()

    T, D = q.shape[-2:]
    scale = D**-0.5
    outputs = []

    for i in range(0, T, block_q):
        # 对T维度做分片
        qi = q[..., i : i + block_q, :]

        # 记录softmax的最大值？
        # [B, H, T, 1]
        m = qi.new_full((*qi.shape[:-1], 1), -float("inf"))
        # 记录softmax的指数和
        l = qi.new_zeros((*qi.shape[:-1], 1))
        # 尚未归一化的加权和
        acc = torch.zeros_like(qi)
        for j in range(0, T, block_k):
            # [B, H, block_k, D]
            kj = k[..., j : j + block_k, :]
            vj = v[..., j : j + block_k, :]

            scores = (qi @ kj.transpose(-2, -1)) * scale

            m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))

            # 旧状态从以 m 为基准，换成以 m_new 为基准
            alpha = torch.exp(m - m_new)

            # 当前块也使用同一个基准
            weights = torch.exp(scores - m_new)

            l = alpha * l + weights.sum(dim=-1, keepdim=True)
            acc = alpha * acc + weights @ vj
            m = m_new

        outputs.append(acc / l)

    return torch.cat(outputs, dim=-2).to(original_dtype)


class FusionMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, device: torch.device, block_size: tuple[int, int] = (32, 32)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_q = nn.Linear(dim, dim, device=device)
        self.proj_k = nn.Linear(dim, dim, device=device)
        self.proj_v = nn.Linear(dim, dim, device=device)
        self.block_size = block_size

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        T = x.size(1)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # [B, T, C] => [B, T, H ,D] => [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        o = flash_attention(q, k, v, block_size=self.block_size)
        o = o.transpose(1, 2).contiguous().view(B, T, self.dim)
        return o


def diff_result():
    torch.manual_seed(0)
    x = torch.randn(5, 128, 256, device="cuda")
    # Baseline
    torch_mha = torchMHA(256, 8).to(device=x.device, dtype=x.dtype)

    normal_mha = NormalMHA(256, 8).to(device=x.device, dtype=x.dtype)
    fusion_mha = FusionMHA(256, 8, device=x.device)
    normal_mha.load_state_dict(torch_mha.state_dict())
    fusion_mha.load_state_dict(torch_mha.state_dict())

    with torch.inference_mode():
        t_o = torch_mha(x)
        t_n = normal_mha(x)
        f_o = fusion_mha(x)

    abs_diff = (t_o - t_n).abs()
    rtol, atol = 1e-5, 1e-6
    # 判断两个张量中是否所有元素都在误差允许范围内相等
    is_close_n = torch.allclose(t_o, t_n, rtol=rtol, atol=atol)
    is_close_f = torch.allclose(t_o, f_o, rtol=rtol, atol=atol)
    print(f"output shape: {tuple(t_o.shape)}")
    print(f"max abs diff: {abs_diff.max().item():.8e}")
    print(f"mean abs diff: {abs_diff.mean().item():.8e}")
    print(f"FusionMHA max abs diff: {(t_o - f_o).abs().max().item():.8e}")
    print(f"allclose (rtol={rtol}, atol={atol}): N={is_close_n}, F={is_close_f}")

    torch.testing.assert_close(t_o, t_n, rtol=rtol, atol=atol)
    torch.testing.assert_close(t_o, f_o, rtol=rtol, atol=atol)


def benchmarking(seq_len: int, dim: int, block_size: tuple[int, int] = (32, 32)):
    x = torch.randn(5, seq_len, dim, device="cuda")
    torch_mha = torchMHA(dim, 8).to(device=x.device, dtype=x.dtype)
    normal_mha = NormalMHA(dim, 8).to(device=x.device, dtype=x.dtype)
    fusion_mha = FusionMHA(dim, 8, device=x.device, block_size=block_size)

    def run_t():
        with torch.inference_mode():
            torch_mha(x)

    def run_n():
        with torch.inference_mode():
            normal_mha(x)

    def run_f():
        with torch.inference_mode():
            fusion_mha(x)

    # profile("FusionMHA", run_f)
    # profile("torchMHA", run_t)

    benchmark(f"torchMHA_{seq_len}_{dim}_{block_size}", run_t)
    benchmark(f"FusionMHA_{seq_len}_{dim}_{block_size}", run_f)

    # Benchmark[torchMHA]: mean: 77.094 ms, std: 0.014 ms
    # Benchmark[NormalMHA]: mean: 300.450 ms, std: 0.323 ms
    # Benchmark[FusionMHA]: mean: 82.987 ms, std: 0.603 ms


if __name__ == "__main__":
    diff_result()
    block_size_list = [16, 32, 64, 128]
    for block_size in block_size_list:
        benchmarking(8196, 512, (block_size, block_size))
        print("============")
