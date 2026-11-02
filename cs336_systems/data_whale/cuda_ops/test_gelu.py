


import os
import torch
import torch.nn.functional as F
# CUDA是C/C++ 扩展，可以直接调用GPU API, 需要启动c++扩展
from torch.utils.cpp_extension import load_inline

from cs336_systems.data_whale.profile_tool import benchmark, profile

# 设置调试模式，这样如果出现错误，CUDA 会告诉你出了什么问题。
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def create_cuda_gelu():
    gelu_path = os.path.join(os.path.dirname(__file__), "gelu.cu")
    build_directory = os.path.join(os.path.dirname(__file__), "dist")
    os.makedirs(build_directory, exist_ok=True)
    # 读到的是源码
    cuda_gelu_src = open(gelu_path).read()
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"
    
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory=build_directory
    )
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu


def profile_gelu():
    cuda_gelu = create_cuda_gelu()
    input_x = torch.randn(1024, 256, device='cuda')
    def run_cuda_gelu():
        cuda_gelu(input_x)

    def run_pytorch_gelu():
        torch.nn.functional.gelu(input_x, approximate="tanh")

    def manual_gelu():
        result = 0.5 * input_x * (1 + torch.tanh(0.79788456 * (input_x + 0.044715 * input_x * input_x * input_x)))
        return result

    compile_manual_gelu = torch.compile(manual_gelu)


    benchmark("manual_gelu", manual_gelu)
    benchmark("compile_manual_gelu", compile_manual_gelu)
    benchmark("cuda_gelu", run_cuda_gelu)
    benchmark("pytorch_gelu", run_pytorch_gelu)



if __name__ == "__main__":
    profile_gelu()
