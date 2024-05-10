from vllm import LLM, SamplingParams
LLM(model="meta-llama/Meta-Llama-3-70B-Instruct",
    gpu_memory_utilization=0.95,
    tensor_parallel_size=4,
    swap_space=16)