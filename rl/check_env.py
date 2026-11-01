import torch, vllm
from vllm import LLM, SamplingParams

def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| vllm", vllm.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("matmul ok:", (x @ x).float().sum().item())

    llm = LLM("HuggingFaceTB/SmolLM2-135M-Instruct", gpu_memory_utilization=0.2)
    out = llm.generate(["What color is a banana?"], SamplingParams(max_tokens=16))
    print("vllm ok:", out[0].outputs[0].text)

if __name__ == "__main__":
    main()
