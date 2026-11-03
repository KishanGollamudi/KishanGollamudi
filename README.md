⚡ Cloud-Native LLM Serving on Google Cloud TPU v5e

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" width="100%" height="auto" style="border-radius: 14px; margin: 20px 0; max-width: 880px;">
  <!-- Background Canvas -->
  <rect width="880" height="340" fill="#221a0f" rx="14" stroke="#a57a4c" stroke-width="1.5" stroke-opacity="0.35" />

<!-- Metric 1 -->
<text x="135" y="38" fill="#a57a4c" font-family="'Inter', -apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle">COLD GRAPH COMPILE</text>
<text x="135" y="68" fill="#dc3958" font-family="'Inter', -apple-system, sans-serif" font-size="22" font-weight="700" text-anchor="middle">26.84s</text>

<!-- Metric 2 -->
<text x="400" y="38" fill="#a57a4c" font-family="'Inter', -apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle">WARM SINGLE INFERENCE</text>
<text x="400" y="68" fill="#f79a32" font-family="'Inter', -apple-system, sans-serif" font-size="22" font-weight="700" text-anchor="middle">6.38s</text>

<!-- Metric 3 -->
<text x="665" y="38" fill="#a57a4c" font-family="'Inter', -apple-system, sans-serif" font-size="11" font-weight="600" text-anchor="middle">BATCH THROUGHPUT</text>
<text x="665" y="68" fill="#889b4a" font-family="'Inter', -apple-system, sans-serif" font-size="22" font-weight="700" text-anchor="middle">38.1 tok/s</text>


🧭 Interactive Deep Dives & Systems Engineering

Unlike traditional NVIDIA GPU architectures that rely on SIMT (Single Instruction, Multiple Threads) where thousands of independent ALUs fetch and write intermediate calculations from registers and high-latency VRAM caches, the Google Cloud TPU v5e runs on a specialized Matrix Multiply Unit (MXU) wired as a 2D Systolic Array:

Continuous Lockstep Dataflow: Activations stream continuously through hardwired multiply-accumulate cells while model weights remain pinned in the execution register grid. Intermediate dot-products flow directly across adjacent cells without making round-trips to High Bandwidth Memory (HBM).

Native Precision Geometry: Execution operates natively in bfloat16. Retaining an 8-bit dynamic range identical to standard IEEE FP32 while maintaining a 16-bit footprint prevents underflow and overflow in deep transformer layers without loss scaling.

Arithmetic Intensity Scaling: Because dense matrix multiplication executes simultaneously across the MXU grid, increasing batch size from 1 to 4 resulted in an identical wall-clock runtime (~6.3s), driving effective serving throughput from 6.4 tok/s to 38.1 tok/s.

During initial batch generation, the model yielded corrupted, repetitive tokens (In,**wen. a a large Cloud...). Debugging revealed that standard Hugging Face tokenizers default to right-padding (padding_side='right'):

❌ Right-Padding (Corrupted Causal Mask):
[Prompt Token 1] [Prompt Token 2] [PAD] [PAD] ---> [Target Generation Boundary]
                                     ^--- Causal mask M zeroed out at boundary


In autoregressive decoders, token prediction occurs at index $N+1$ directly following the final sequence element. With right-padding, that boundary lands on a [PAD] token whose attention weight is masked out ($-\infty$), degrading the self-attention calculation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

✅ Strict Left-Padding (Stabilized Attention):
[PAD] [PAD] [Prompt Token 1] [Prompt Token 2] ---> [Target Generation Boundary]
                                              ^--- Attention state fully intact


By enforcing tokenizer.padding_side = 'left', input sequences sat flush against the generation boundary, restoring coherent attention distribution and clean text generation.

PyTorch-XLA does not execute commands in eager mode. Operations are traced into an intermediate Directed Acyclic Graph (DAG):

Cold-Start Latency (26.84s): Upon receiving the first token generation request, OpenXLA intercepts the entire computational graph of Qwen2.5-Coder-1.5B, executes layer fusion, allocates static addresses across the 16 GB HBM, and compiles the graph into machine instructions.

Warm Execution (6.38s): After compilation, the execution plan is cached in memory. Subsequent requests with matching tensor dimensions execute directly on the hardware.

Production Static Bucketing: Dynamic sequence lengths cause XLA to invalidate the cached graph and trigger recompilation cycles (~41s). In production pipelines, incoming batches are aligned into fixed buckets (e.g., lengths of 128, 256, 512, 1024) to constrain compiled graphs to a finite set.

📊 Benchmark Latency & Throughput Profile

Execution Phase

Batch Size

Latency

Effective Throughput

Hardware State

Cold Graph Build

1

26.84s

~4.5 tok/s

Initial OpenXLA AOT compilation & HBM memory allocation

Warm Single Run

1

6.38s

6.4 tok/s

Cached XLA graph execution on TPU v5e core

Dense Systolic Batch

4

6.30s

38.1 tok/s

Reused model weights across parallel matrix dimensions

🗂️ Repository Architecture

.
├── tpu_llm_serving_pipeline.ipynb   # Verified TPU v5e execution notebook
├── app.py                           # Gradio UI & FastAPI inference engine
├── requirements.txt                 # OpenXLA, PyTorch, Transformers, Gradio dependencies
└── README.md                        # Architecture documentation & benchmarks


🚀 Step-by-Step Execution Walkthrough

1. Initialize Hardware Verification

import torch

# Connect to native TPU runtime accelerator
device = torch.device("tpu")
print(f"Active Execution Silicon: {device}")

# Verify systolic array matrix multiplication
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)
z = torch.matmul(x, y)
print("Hardware tensor verification passed:\n", z)


2. Configure Model & Left-Padding Tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Critical: Strict left-padding prevents causal decoder mask distortion
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Allocate weights directly in native bfloat16 into 16 GB HBM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device)


3. Launch Interactive Gradio Dashboard

import gradio as gr

def chat_tpu(message, history, max_new_tokens, temperature):
    conversation = []
    for turn in history:
        if isinstance(turn, dict):
            conversation.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            if turn[0]: conversation.append({"role": "user", "content": turn[0]})
            if turn[1]: conversation.append({"role": "assistant", "content": turn[1]})
    conversation.append({"role": "user", "content": message})
    
    formatted_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0.0),
            temperature=float(temperature) if temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

demo = gr.ChatInterface(
    fn=chat_tpu,
    title="Cloud TPU v5e-1 Inference Dashboard",
    additional_inputs=[
        gr.Slider(minimum=32, maximum=512, value=256, step=16, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
    ]
)

demo.launch(share=True)


Engineered by Kishan Gollamudi

Cloud, DevOps & MLOps Infrastructure Engineer
