#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    // 通过block坐标，拿到当前线程需要计算的元素索引i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 越界检查
    // if (i >= num_elements) {
    //     return;
    // }
    if (i < num_elements) {
        // 原来的实现，没有显示地声明数据类型，所以在计算的过程中，部分计算会被转成double，我的L4显卡对float63的计算性能比较弱，所以实现出来的性能不好
        // out[i] = 0.5 * in[i] * (1 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));

        float x_in = in[i];
        out[i] = 0.5f * x_in * (1 + tanhf(0.79788456f * (x_in + 0.044715f * x_in * x_in * x_in)));
    }
    
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// 第二部分 gelu 函数

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());
    torch::Tensor y = torch::empty_like(x);
    int num_elements = x.numel();
    int block_size = 1024;
    int num_blocks = cdiv(num_elements, block_size);
    
    //启动内核
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
