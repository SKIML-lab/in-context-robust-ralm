from openai import OpenAI
from typing import List
from vllm import LLM, SamplingParams

LLAMA3_CHAT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
               "You are a helpful assistant who follows instructions well.<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n")
QWEN_CHAT = ("<|im_start|>system\nYou are a helpful assistant who follows instructions well.<|im_end|>\n"
            "<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n")

def gpt_chat_completion(prompt: str, args, client = None):
    if not client:
        client = OpenAI()
    cnt = 0
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            seed=42,
            temperature=0,
            max_tokens=args.max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions well."},
                {"role": "user", "content": prompt},
            ]
            )
            break
        except Exception as e:
            print(e)
            print("Failed to get response")
            if cnt >= 3:
                return None
            cnt += 1
    return response.choices[0].message.content.strip()

def vllm_completion(prompts: List[str], args) -> List[str]:
    import torch
    llm = LLM(model=args.llm,
              gpu_memory_utilization= 0.95,
              max_context_len_to_capture=4096,
              seed=42,
              max_model_len=4096,
              swap_space=4,
              tensor_parallel_size=torch.cuda.device_count())
    config = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    if "llama" in args.llm.lower():
        prompts = [LLAMA3_CHAT.format(PROMPT=p) for p in prompts]
    else:
        prompts = [QWEN_CHAT.format(PROMPT=p) for p in prompts]
    output= llm.generate(prompts=prompts, sampling_params=config)
    output = [o.outputs[0].text.strip() for o in output]
    if "llama" in args.llm.lower():
        output = [o.split("<|eot_id|>")[0] for o in output]
    return output