# Problem (benchmarking_script):  Benchmarking Script (4 points)
(a) Write a script to perform basic end-to-end benchmarking of the `forward pass`, `backward pass`, and `optimizer step` in your model. Specifically, your script should support the following:
- Given hyperparameters (e.g., number of layers), initialize a model.
- Generate a random batch of data
- Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛
steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python timeit module (e.g., either using the timeit function, or using timeit.default_timer(), which gives you the system’s highest resolution clock, thus a better default for benchmarking than time.time()).
- Call torch.cuda.synchronize() after each step.
Deliverable: A script that will initialize a basics Transformer model with the given
hyperparameters, create a random batch of data, and time forward-only, forward-andbackward, and full training steps that include the optimizer step.
[Deliver]: benchmark.py

(b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?
Deliverable: A 1-2 sentence response with your timings.
我的是一张Nvidia L4显卡，显存比较少
| step type | model size | mean time (ms) | std dev (ms) |
| --- | --- | --- | --- |
|forward|small|100.693|1.539|
|forward_backward|small|309.522|2.399|
|full|small|347.998|1.117|
|forward|medium|298.763|1.976|
|forward_backward|medium|910.515|2.197|
|full|medium|1043.634|1.608|
|forward|large|648.690|1.858|
|forward_backward|large|1937.879|5.337|
|full|large|OOM|OOM|
|forward|xl|1939.954|1.030|
|forward_backward|xl|OOM|OOM|
|full|xl|OOM|OOM|
|forward|10B|OOM|OOM|
|forward_backward|10B|OOM|OOM|
|full|10B|OOM|OOM|


(c)One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?
Deliverable: A 2-3 sentence response
[Deliver]: Warmup是为了预热GPU，避免前几次执行因为Kernel初始化、内存-显存装载、GPU频率调整等原因，干扰测试结果


# Problem (nsys_profile):  Nsight Systems Profiling (5 points)
Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:
(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?
Deliverable: A 1-2 sentence response.
[Deliver]:
执行命令：
```bash
uv run nsys profile -o small_forward --trace=cuda python  cs336_systems/a_2_profile_and_benchmark/benchmark.py --mode forward --size small
```
forward总耗时5.2秒，比benchmark测量的耗时要多很多

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)
Deliverable: A 1-2 sentence response
[Deliver]:
执行耗时最长的kernel是ampere_sgemm_128x64_tn， 耗时占比kernel总耗时的51.2%, 执行了1275次
|Time (%) | Total Time (ns) | Instances  Avg (ns) |  Med (ns) |  Min (ns) | Max (ns) |  StdDev (ns) | Name |
|51.2 |     757,122,496   |   1,275   |   593,821.6 |   267,966.0 |   179,551 |   3,257,796 |   451,246.4 | ampere_sgemm_128x64_tn    |
其他人kernel都小于 6.2%

接下来执行forward_backward
```bash
uv run nsys profile -o small_forward_backward --trace=cuda,nvtx python  cs336_systems/a_2_profile_and_benchmark/benchmark.py --mode forward_backward --size small
```
forward_backward里面，耗时最长的kernel还是这个，但是ampere_sgemm_128x64_nn的耗时快追上来了
|Time (%) | Total Time (ns) | Instances  Avg (ns) |  Med (ns) |  Min (ns) | Max (ns) |  StdDev (ns) | Name |
|15.9 | 722,649,507 |   1,275 | 566,783.9 | 265,278.0 | 180,159 | 3,452,642 | 443,064.9  | ampere_sgemm_128x64_tn|
|14.8 | 675,323,917 |   1,275 | 529,665.8 | 246,398.0 | 184,447 | 3,458,146 | 422,839.3  | ampere_sgemm_128x64_nn |

(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
[Deliver]:
at::native::vectorized_elementwise_kernel 占了 12.8%
void cutlass::Kernel2 占了 9.6%

(d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
Deliverable: A 1-2 sentence response.

(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?
Deliverable: A 1-2 sentence response.

# Problem (mixed_precision_accumulation):  Mixed-Precision Accumulation (1 point)
Run the following code and comment on the accuracy of the results.
```python
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```
[Deliver]:
这题是想让我感受，不同精度下，累加的精度损失不同
tensor(10.0001) # Float32 损失最小
tensor(9.9531, dtype=torch.float16) # FP16 损失最大，因为表示精度的bit最少
tensor(10.0021) # 输入FP16，输出Float32，损失会小很多
tensor(10.0021) # 输入FP16，在计算的时候转成FP32，跟直接输入FP16没有区别
第三和第四的损失是一样的，题目希望我理解：混合精度可以减小计算过程中的累积误差，但是没法弥补输入精度的损失

# Problem (benchmarking_mixed_precision):  Benchmarking Mixed Precision (2 points)
(a) Consider the following model:
```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```
Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
• the model parameters within the autocast context?
• the output of the first feed-forward layer (ToyModel.fc1)?
• the output of layer norm (ToyModel.ln)?
• the model’s predicted logits?
• the loss?
• the model’s gradients?
Deliverable: The data types for each of the components listed above.
[Deliver]:
- 模型参数都是BF16 - 这题错了：AMP是计算时使用FP16，它不影响参数类型本身
- output of fc1是FP16
- output of layer norm是FP32
- predicted logits是FP16
- loss是FP32
- gradients是FP16 - 跟第一题一样


(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?
Deliverable: A 2-3 sentence response
[Deliver]:
因为在做归一化的时候，需要处理大量0-1之间的小数，如果精度不够的话，误差会比较大
> ↑这里不太对，不能用FP16，是因为在norm的时候有均值、方差等累加操作，容易堆积误差
如果改用BF16，仍然需要，因为BF16的精度更低

(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in Section 2.1.2. Compare the results of using full precision versus mixed precision, and comment on any trends as model size changes. You may find the nullcontext no-op context manager to be useful.
Deliverable: A 2-3 sentence response with your timings and commentary
[Deliver]:
|forward|small|bf16|45.474|0.082|
|forward|small|no-bf16|100.471|1.729|
|forward|medium|bf16|139.415|0.715|
|forward|medium|no-bf16|298.860|2.245|
|forward|large|bf16|288.376|0.360|
|forward|large|no-bf16|648.477|3.096|
|forward_backward|small|bf16|152.805|0.382|
|forward_backward|small|no-bf16|308.855|2.595|
|forward_backward|medium|bf16|453.113|0.707|
|forward_backward|large|bf16|926.750|0.615|
|forward_backward|large|no-bf16|1944.412|4.888|
|forward_backward|medium|no-bf16|910.799|2.656|
BF16的优化结果，没有随着模型尺寸的增加而增加，而是稳定在2~2.2倍左右


# Problem (memory_profiling):  Memory Profiling (4 points)
Profile your complete training step of forward pass, backward pass, and optimizer step of the xl model from Table 1 with context lengths of 128 and 2048.

(a) Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the xl model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see? 
Deliverable: Two images of the “Active memory timeline” of an xl model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.
[Deliver]:
```bash
uv run cs336_systems/a_2_profile_and_benchmark/benchmark.py --bf16 --memory-profile --mode forward --size xl --context-length 2048 --batch-size 2
```
128 + forward：
Allocated:      13132.80 MiB
Reserved:       13216.00 MiB
Peak allocated: 13205.93 MiB
Peak reserved:  13216.00 MiB

128 + full: OOM

2048 + forward:
Allocated:      13133.45 MiB
Reserved:       17524.00 MiB
Peak allocated: 16429.96 MiB
Peak reserved:  17524.00 MiB

2048 + full: OOM
内容长度增加，会显著增加被使用到的内存，但不会改变模型所占用的内存本身

(b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?
Deliverable: A table with two numbers per context length.
这道题想让我观察到：训练时的内存占用要高于推理，因为autograd和优化器都很占内存


(c) Find the peak memory usage of the xl model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?
Deliverable: A 2-3 sentence response.