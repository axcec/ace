---
title: "LLM Inference Deep Dive — KV Cache, Batching, Memory, Latency & Cost"
date: 2026-08-16
---

# LLM Inference Deep Dive — KV Cache, Batching, Memory, Latency & Cost

Running an LLM in production is not simply a matter of loading a model onto a GPU and calling `generate()`.

At small scale, inference looks deceptively simple:

```text
Prompt → Model → Response
```

At production scale, the system looks very different:

```text
                    Requests
                       │
                       ▼
                ┌─────────────┐
                │ API Gateway │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │   Queue     │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │  Scheduler  │
                └──────┬──────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Prefill             Decode
              │                 │
              └────────┬────────┘
                       ▼
                  KV Cache
                       │
                       ▼
                     GPU
                       │
                       ▼
                Token Stream
```

Now the engineering questions become much more interesting:

* How much GPU memory does the model require?
* How much additional memory does the KV cache consume?
* How many requests can a GPU handle concurrently?
* Why does batching improve throughput?
* Why can batching also increase latency?
* Why is a long context expensive even when the model itself hasn't changed?
* What determines time-to-first-token?
* When should we use quantization?
* How do inference engines such as vLLM optimize these workloads?
* How do we minimize cost per generated token?

This is the domain of **LLM inference engineering**.

---

## 1. What Is LLM Inference?

Inference is the process of using a trained model to generate predictions.

For an autoregressive language model, this means predicting the next token based on all previously seen tokens.

Given:

```text
The capital of France is
```

the model estimates:

```text
P(token | "The capital of France is")
```

and may predict:

```text
Paris
```

The new sequence becomes:

```text
The capital of France is Paris
```

The model then predicts the next token again.

This process continues until the model generates an end-of-sequence token or reaches the configured generation limit.

Conceptually:

```text
Prompt
  │
  ▼
Tokenization
  │
  ▼
Transformer
  │
  ▼
Next Token
  │
  ▼
Append Token
  │
  ▼
Transformer
  │
  ▼
Next Token
  │
  ▼
...
```

This sequential nature is one of the fundamental challenges of LLM inference.

A model cannot simply generate every output token independently because token `N+1` depends on token `N`.

---

# 2. Prefill and Decode

A useful way to understand inference is to divide it into two phases:

```text
             LLM Inference
                  │
          ┌───────┴───────┐
          │               │
       Prefill          Decode
          │               │
   Process prompt     Generate tokens
```

## Prefill

During prefill, the model processes the input prompt.

For example:

```text
You are a helpful assistant.
Explain distributed systems.
```

The model processes all these input tokens and computes their representations.

This phase is highly parallelizable because the input tokens already exist.

The output of this computation is used to populate the KV cache.

---

## Decode

After prefill, the model begins generating tokens.

For example:

```text
Distributed
systems
are
systems
...
```

Each new token depends on the previous tokens.

Therefore, decode is inherently sequential.

This distinction is extremely important when analyzing performance.

A workload with:

```text
10,000 input tokens
+
100 output tokens
```

behaves very differently from:

```text
100 input tokens
+
10,000 output tokens
```

Even though both requests contain approximately the same number of total tokens.

---

# 3. Why the KV Cache Exists

The most important optimization in autoregressive inference is the **Key-Value cache**, commonly called the **KV cache**.

To understand why it exists, we first need to look at attention.

Transformer attention uses queries, keys and values:

```text
Q = Query
K = Key
V = Value
```

During generation, the model needs information from previously generated tokens.

Without caching, the model would repeatedly recompute the keys and values associated with previous tokens.

For a sequence like:

```text
Token 1
Token 2
Token 3
Token 4
Token 5
```

the naive approach would repeatedly process the history.

Instead, inference systems store the previously calculated key/value tensors:

```text
Token 1 ──┐
Token 2 ──┤
Token 3 ──┼──► KV Cache
Token 4 ──┤
Token 5 ──┘
```

When the next token is generated, the model can reuse this cached information.

This dramatically reduces redundant computation.

But there is a trade-off:

> **The KV cache consumes GPU memory.**

And unlike model weights, which are mostly fixed, KV-cache usage changes dynamically with workload.

---

# 4. Understanding KV-Cache Memory

A simplified KV-cache memory calculation looks like:

```text
KV memory ≈
2 × number_of_layers
× number_of_KV_heads
× head_dimension
× sequence_length
× bytes_per_element
× number_of_sequences
```

The `2` exists because we store both:

```text
K = Keys
V = Values
```

Consider a simplified model with:

```text
Layers:          32
KV heads:        8
Head dimension: 128
Context:         4096 tokens
Datatype:        FP16
```

FP16 requires:

```text
2 bytes
```

per element.

The approximate KV-cache memory for one sequence is:

```text
2 × 32 × 8 × 128 × 4096 × 2 bytes
```

which is roughly:

```text
512 MiB
```

for this simplified configuration.

Now imagine 32 concurrent requests:

```text
512 MiB × 32 ≈ 16 GiB
```

The model weights haven't changed.

The model architecture hasn't changed.

But the workload has consumed another large chunk of GPU memory.

This is why **concurrency and context length are infrastructure variables**.

---

# 5. GPU Memory Is More Than Model Weights

A common mistake is to calculate only the model's weight memory.

Suppose a GPU has:

```text
80 GB VRAM
```

and the model weights require:

```text
50 GB
```

It is tempting to assume:

```text
80 - 50 = 30 GB available
```

But production inference needs memory for much more than weights.

A simplified memory model is:

```text
GPU Memory
│
├── Model Weights
├── KV Cache
├── Activations
├── CUDA / Runtime Memory
├── Temporary Buffers
└── Communication Buffers
```

Therefore:

```text
Available KV Cache
=
GPU Memory
-
Model Memory
-
Runtime Overhead
-
Other Allocations
```

If the KV cache grows too large, the system may no longer be able to admit new requests.

That makes GPU memory a scheduling constraint.

---

# 6. Why Long Contexts Are Expensive

Consider two requests:

```text
Request A → 1,000 tokens
Request B → 32,000 tokens
```

The second request isn't merely 32 times more text from an application perspective.

It also requires substantially more state to be maintained during inference.

Long contexts can lead to:

```text
Longer context
      ↓
Larger KV cache
      ↓
Higher memory usage
      ↓
Lower concurrency
      ↓
Potentially lower throughput
```

This is one reason context-window size is not simply a model capability.

A model may support a very large context window, but serving many long-context requests simultaneously can become extremely expensive.

---

# 7. Batching

If a GPU processes one request at a time:

```text
GPU
 │
 ├── Request A
 │
 ├── Request B
 │
 ├── Request C
 │
 └── Request D
```

the GPU may spend a significant amount of time underutilized.

GPUs are designed to execute enormous amounts of parallel computation.

Batching allows multiple requests to be processed together:

```text
             Batch
        ┌────┼────┐
        ▼    ▼    ▼
       R1   R2   R3
        └────┼────┘
             ▼
            GPU
```

This can significantly increase throughput.

But traditional batching has a problem.

---

# 8. The Problem With Static Batching

Suppose we have four requests:

```text
R1 → 20 output tokens
R2 → 100 output tokens
R3 → 500 output tokens
R4 → 2000 output tokens
```

If the system waits for every request in the batch to finish before starting another batch, the shorter requests effectively finish early while the GPU continues processing the longest request.

Conceptually:

```text
R1  ████
R2  █████████
R3  █████████████████
R4  █████████████████████████████████
```

The batch remains alive because of R4.

This is inefficient.

---

# 9. Continuous Batching

Modern inference engines use **continuous batching** or **in-flight batching** to address this problem.

Instead of treating a batch as a fixed group, requests can dynamically enter and leave the running workload.

For example:

```text
Time →

R1 ─────────X
R2 ─────────────────X
R3 ───────X
             R5 ───────────X
R4 ───────────────────────X
```

When R3 finishes, another request can take its place.

The GPU can therefore remain highly utilized.

This is one of the major differences between a simple model-serving implementation and a production inference engine.

---

# 10. Throughput vs Latency

Inference optimization is full of trade-offs.

Increasing batch size can improve throughput:

```text
Batch size ↑
     ↓
GPU utilization ↑
     ↓
Throughput ↑
```

But it can also increase latency:

```text
More requests
     ↓
More scheduling / queueing
     ↓
Higher latency
```

Therefore, maximizing throughput is not always the right objective.

Suppose an application has an SLO:

```text
TTFT < 500 ms
```

A configuration that provides:

```text
10,000 tokens/sec
```

but causes:

```text
TTFT = 2 seconds
```

may be unacceptable for an interactive product.

Inference engineering is therefore about finding the right operating point between:

```text
Latency
Throughput
Cost
```

---

# 11. Time to First Token

For streaming applications, users don't necessarily care only about total response time.

They care about:

> **How quickly does the response start?**

This is measured using **Time to First Token (TTFT)**.

```text
Request
│
│  processing
│
│  processing
│
▼
First token
```

A request might take 5 seconds overall but still feel responsive if the first token appears after 200 ms.

Conversely, a 2-second response can feel slow if nothing appears for the first 1.5 seconds.

TTFT is influenced by:

* queueing
* prompt length
* prefill computation
* batching
* GPU utilization
* scheduler behavior
* cache availability

---

# 12. Time Per Output Token

After the first token arrives, another important metric is the rate at which subsequent tokens are generated.

This is often expressed as:

**Time Per Output Token (TPOT)**

For example:

```text
First token: 300 ms

Token 2: 30 ms
Token 3: 31 ms
Token 4: 29 ms
Token 5: 30 ms
```

The user experiences:

```text
wait...
→ token
→ token
→ token
→ token
```

So a useful inference profile contains both:

```text
TTFT → responsiveness
TPOT → generation speed
```

---

# 13. PagedAttention

One of the challenges with KV caching is memory management.

Traditional allocation strategies can cause fragmentation because requests have different sequence lengths and lifetimes.

A useful solution is to divide the KV cache into blocks.

Conceptually:

```text
Logical sequence:

[A][B][C][D]

Physical GPU memory:

[C]
[A]
[D]
[B]
```

The logical sequence doesn't need to occupy one contiguous region of GPU memory.

This approach is known as **PagedAttention**.

It allows inference systems to manage KV-cache memory more efficiently and dynamically.

This idea was one of the key innovations behind vLLM.

The important conceptual shift is:

> The inference engine isn't just running the model. It is managing a dynamic memory system specifically for transformer inference.

---

# 14. Prefix Caching

Many real-world applications repeatedly send the same prefix.

For example:

```text
System Prompt
+
Company Policies
+
Tool Instructions
+
User Query
```

Only the final user query changes.

Without caching:

```text
Request 1 → compute entire prefix
Request 2 → compute entire prefix
Request 3 → compute entire prefix
Request 4 → compute entire prefix
```

With prefix caching:

```text
Shared Prefix
      │
      ▼
Compute once
      │
      ▼
Reuse cached KV blocks
      │
      ├── Request A
      ├── Request B
      ├── Request C
      └── Request D
```

This can reduce redundant prefill computation.

Prefix caching can be particularly useful for:

* AI agents
* RAG systems
* enterprise assistants
* long system prompts
* multi-turn conversations

---

# 15. Quantization

Another major inference optimization is **quantization**.

Model weights are commonly stored using formats such as:

```text
FP32
FP16
BF16
FP8
INT8
INT4
```

Lower-precision representations can reduce memory usage.

For example:

```text
FP16 → 16 bits
INT8 → 8 bits
INT4 → 4 bits
```

Ignoring implementation details and overhead, moving from FP16 to INT8 can approximately halve the storage required for the weights.

The benefits can include:

```text
Lower memory
      ↓
Larger effective batch
      ↓
Higher concurrency
      ↓
Potentially higher throughput
```

But quantization is not free.

Depending on the method and model, there can be trade-offs in:

* model quality
* numerical stability
* kernel performance
* hardware compatibility

The correct question isn't:

> "Should I quantize?"

It is:

> "What precision gives me the best quality/cost/latency trade-off for this workload?"

---

# 16. Speculative Decoding

Autoregressive decoding is sequential:

```text
Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
Token 4
```

Speculative decoding tries to reduce this bottleneck.

A smaller draft model predicts several candidate tokens:

```text
Small model

Token 1 → Token 2 → Token 3 → Token 4
```

The larger model then verifies those candidates.

Conceptually:

```text
             Draft Model
                  │
                  ▼
          Candidate Tokens
                  │
                  ▼
            Target Model
                  │
          ┌───────┴───────┐
          ▼               ▼
       Accept            Reject
```

If the target model accepts many of the proposed tokens, multiple tokens can effectively be advanced in fewer expensive target-model steps.

This can improve decoding performance for suitable workloads.

---

# 17. Chunked Prefill

Long prompts introduce another scheduling problem.

Imagine one request contains:

```text
100,000 tokens
```

If the inference engine processes the entire prompt as one huge prefill operation, that request can dominate GPU resources.

Other users may have to wait.

Chunked prefill divides the work:

```text
100k tokens

↓
20k
↓
20k
↓
20k
↓
20k
↓
20k
```

The scheduler can then interleave this work with other requests.

This provides finer-grained control over:

```text
Prefill
+
Decode
```

and can help improve latency under mixed workloads.

---

# 18. What Does an Inference Engine Actually Do?

At this point, it should be clear that an inference engine is much more than a wrapper around a model.

A production inference engine may contain:

```text
                    Inference Engine
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    Scheduler        KV Manager       Batch Manager
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                   Model Execution
                          │
                          ▼
                    GPU Kernels
```

Its responsibilities can include:

* request scheduling
* continuous batching
* KV-cache allocation
* memory management
* prefix caching
* quantization
* optimized attention
* speculative decoding
* distributed execution
* streaming
* GPU utilization

This is why systems such as vLLM and TensorRT-LLM are important pieces of the modern LLM infrastructure stack.

---

# 19. Distributed Inference

Eventually, the model may become too large for one GPU.

Suppose:

```text
Model = 400 GB
GPU   = 80 GB
```

The model cannot simply be loaded onto one GPU.

We need distributed execution.

One approach is **tensor parallelism**:

```text
                 Model
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      GPU 0      GPU 1      GPU 2
```

Another is **pipeline parallelism**:

```text
GPU 0
Layers 0–15
   │
   ▼
GPU 1
Layers 16–31
   │
   ▼
GPU 2
Layers 32–47
```

And another is **data parallelism**:

```text
              Requests
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     GPU 0      GPU 1      GPU 2
   Model copy  Model copy  Model copy
```

The correct strategy depends on:

* model size
* GPU memory
* network topology
* interconnect bandwidth
* workload
* latency requirements

---

# 20. GPU Communication Becomes Important

With multiple GPUs, compute isn't the only bottleneck.

You also have communication:

```text
GPU 0 ←────────→ GPU 1
       communication
GPU 2 ←────────→ GPU 3
```

If GPUs constantly need to exchange data, communication overhead can limit scaling.

This produces an important lesson:

> More GPUs do not automatically mean proportionally more performance.

The system must balance:

```text
Compute
+
Memory bandwidth
+
GPU communication
+
Scheduling
```

This is where LLM inference begins to look increasingly like distributed systems engineering.

---

# 21. Measuring Inference Performance

You cannot optimize what you don't measure.

A basic benchmark should capture:

```text
TTFT
TPOT
Tokens/sec
Requests/sec
GPU utilization
GPU memory utilization
KV-cache utilization
Queue time
Error rate
```

A useful benchmark might look like:

```text
Model:          8B
Precision:      FP16
GPU:            H100
Concurrency:    1, 4, 8, 16, 32
Input tokens:   1k
Output tokens:  256
```

Then compare:

```text
Concurrency | TTFT | TPOT | Throughput | GPU Memory
------------|------|------|------------|-----------
1           | ...  | ...  | ...        | ...
4           | ...  | ...  | ...        | ...
8           | ...  | ...  | ...        | ...
16          | ...  | ...  | ...        | ...
32          | ...  | ...  | ...        | ...
```

This is much more useful than saying:

> "Model X generates 100 tokens/sec."

Performance is workload-dependent.

---

# 22. Cost Per Token

Inference eventually becomes an economics problem.

Imagine a GPU costs:

```text
$3/hour
```

and your serving stack produces:

```text
1,000,000 tokens/hour
```

Ignoring other costs:

```text
$3 / 1,000,000 tokens
```

or:

```text
$0.000003/token
```

But real infrastructure costs include:

```text
GPU
+
CPU
+
RAM
+
Storage
+
Networking
+
Load balancing
+
Observability
+
Idle capacity
```

More importantly, utilization matters.

Suppose the GPU is capable of:

```text
1M tokens/hour
```

but your workload only uses:

```text
300k tokens/hour
```

You are still paying for the GPU.

Therefore:

> **GPU utilization is directly connected to inference economics.**

---

# 23. Why Batching Affects Cost

Suppose you have:

```text
GPU cost = $3/hour
```

Scenario A:

```text
100k tokens/hour
```

Cost:

```text
$3 / 100k
```

Scenario B:

```text
1M tokens/hour
```

Cost:

```text
$3 / 1M
```

The GPU hasn't become cheaper.

The utilization improved.

This is why inference optimization can dramatically reduce cost without changing the model.

Better:

```text
Scheduling
+
Batching
+
Memory utilization
+
Kernel efficiency
```

can translate directly into:

```text
Lower cost per token
```

---

# 24. The Inference Optimization Stack

A useful way to think about optimization is as a stack.

```text
Application
│
├── Reduce unnecessary context
├── Cache repeated prompts
├── Optimize RAG retrieval
│
▼
Serving
│
├── Continuous batching
├── Scheduling
├── Prefix caching
│
▼
Memory
│
├── KV-cache optimization
├── Paged allocation
├── Quantization
│
▼
Model
│
├── Smaller model
├── Distillation
├── Speculative decoding
│
▼
Hardware
│
├── GPU selection
├── Memory bandwidth
├── GPU interconnect
└── Multi-GPU parallelism
```

Optimization at the application layer can sometimes be more valuable than optimization at the GPU layer.

For example, reducing a prompt from:

```text
20k tokens
```

to:

```text
5k tokens
```

can reduce computation and KV-cache requirements without changing the model or hardware.

---

# 25. The Core Trade-offs

LLM inference is fundamentally a collection of trade-offs.

### Larger batch

```text
Throughput ↑
Latency potentially ↑
Memory ↑
```

### Longer context

```text
Context capability ↑
KV memory ↑
Concurrency ↓
```

### Lower precision

```text
Memory ↓
Potential throughput ↑
Potential quality trade-off
```

### More GPUs

```text
Model capacity ↑
Infrastructure cost ↑
Communication overhead ↑
```

### Larger model

```text
Potential quality ↑
Compute ↑
Memory ↑
Cost ↑
```

There is no universal optimal configuration.

The correct configuration depends on the workload.

---

# 26. A Practical Mental Model

When debugging an inference system, ask these questions in order.

### 1. Is the model too large?

```text
Can it fit in GPU memory?
```

### 2. Is the context too large?

```text
How many tokens are we processing?
```

### 3. Is KV cache the bottleneck?

```text
How much memory is concurrency consuming?
```

### 4. Is batching efficient?

```text
Are GPUs actually being utilized?
```

### 5. Is scheduling causing latency?

```text
Are requests spending too much time waiting?
```

### 6. Is compute the bottleneck?

```text
Are GPU compute units saturated?
```

### 7. Is memory bandwidth the bottleneck?

```text
Is the GPU waiting on memory movement?
```

### 8. Is communication the bottleneck?

```text
Are multiple GPUs spending too much time exchanging data?
```

### 9. Can the workload be changed?

```text
Can we reduce context?
Can we cache prefixes?
Can we use a smaller model?
Can we quantize?
```

This way of thinking is more valuable than memorizing individual inference-engine features.

---

# 27. From `model.generate()` to Inference Engineering

A simple implementation might look like:

```python
output = model.generate(
    input_ids,
    max_new_tokens=256
)
```

That is enough to demonstrate that the model works.

But production inference asks very different questions:

```text
How many requests?

What is the concurrency?

What is the average input length?

What is the output length distribution?

How much GPU memory is available?

How much KV cache can we allocate?

What is our TTFT SLO?

What throughput do we need?

How much can we spend per million tokens?
```

The model is only one component.

The serving system determines how efficiently that model can be transformed into a production service.

---

# 28. Conclusion

LLM inference is a systems problem disguised as a machine-learning problem.

The model itself performs the mathematical computation, but production inference depends heavily on everything surrounding it:

```text
                 LLM Inference
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Compute         Memory        Scheduling
       │              │              │
       ▼              ▼              ▼
    GPU kernels    KV Cache       Batching
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                  Throughput
                      │
                      ▼
                    Cost
```

The key concepts are:

* **Prefill** determines how the prompt is processed.
* **Decode** generates tokens sequentially.
* **KV cache** avoids recomputing previous attention state.
* **PagedAttention** improves KV-cache memory management.
* **Continuous batching** improves GPU utilization.
* **Quantization** reduces memory and can improve serving efficiency.
* **Prefix caching** avoids repeated computation.
* **Speculative decoding** can accelerate token generation.
* **Distributed inference** allows models larger than a single GPU.
* **TTFT and TPOT** help measure user-perceived performance.
* **GPU utilization and tokens/sec** determine infrastructure efficiency.
* **Cost per token** connects inference engineering to business economics.

The most important mindset shift is this:

> **LLM inference is not just about running a model. It is about efficiently managing compute, memory, scheduling, and communication under a workload.**

Once you start looking at inference this way, systems like vLLM and TensorRT-LLM become much easier to understand.

They aren't magic performance layers.

They are sophisticated systems designed to answer one fundamental question:

> **How do we extract as much useful work as possible from expensive hardware while keeping latency, memory usage, and cost under control?**

That is the essence of **LLM inference engineering**.
