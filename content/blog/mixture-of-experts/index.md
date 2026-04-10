---
title: "Mixture of Experts: How Modern LLMs Get Smarter Without Getting Slower"
date: 2026-04-10
description: "A walkthrough of Mixture of Experts — from the original 1991 paper to Mixtral and DeepSeek-V2"
tags: ["Mixture of Experts", "Transformers", "LLM Architecture"]
categories: ["Deep Learning"]
math: true
showTableOfContents: false
showHeadingAnchors: true
showReadingTime: true
showAuthor: true
---

{{< katex >}}

Scaling language models has traditionally meant scaling compute proportionally — a model with 2x the parameters requires roughly 2x the FLOPs per token. Mixture of Experts (MoE) breaks this relationship. By training multiple **expert** sub-networks and using a lightweight **router** to select only a few per token, MoE architectures achieve the capacity of a massive model with the per-token compute of a much smaller one.

This is the architecture behind **Mixtral 8x7B** (Jiang et al., 2024), **DeepSeek-V2** (DeepSeek-AI, 2024), and the broader trend toward sparse models in production LLMs. In this post, we walk through the core mechanism, the key papers that shaped MoE, and the practical tradeoffs involved.

---

## The Core Idea

In a standard transformer, every token passes through the **same** feed-forward network (FFN) — the densest and most parameter-heavy component of each layer. Whether the token is a piece of Python code, a French poem, or a calculus equation, it activates the exact same parameters.

MoE replaces that single FFN with **N expert FFNs** and a lightweight **gating network** (the router). For each token, the router scores all experts and picks the top-K. Only those K experts run; the rest stay idle.

{{< mermaid >}}
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px', 'primaryColor': '#14b8a6', 'primaryTextColor': '#fafafa', 'primaryBorderColor': '#0d9488', 'lineColor': '#5eead4', 'secondaryColor': '#27272a', 'tertiaryColor': '#3f3f46', 'noteTextColor': '#a1a1aa', 'noteBkgColor': '#27272a', 'noteBorderColor': '#3f3f46' }}}%%
graph TB
    subgraph DENSE["<b>Dense Transformer Layer</b>"]
        direction TB
        D_IN[Token x] --> D_ATT[Self-Attention]
        D_ATT --> D_FFN["FFN<br/><i>All parameters active</i>"]
        D_FFN --> D_OUT[Output]
        style D_FFN fill:#ef4444,color:#fff,stroke:#dc2626
    end

    subgraph MOE["<b>MoE Transformer Layer</b>"]
        direction TB
        M_IN[Token x] --> M_ATT[Self-Attention]
        M_ATT --> M_ROUTER["Router<br/><i>Scores all experts</i>"]
        M_ROUTER --> M_E1["Expert 1 ✓<br/><i>w = 0.62</i>"]
        M_ROUTER --> M_E2["Expert 2 ✗"]
        M_ROUTER --> M_E3["Expert 3 ✓<br/><i>w = 0.38</i>"]
        M_ROUTER --> M_E4["Expert 4 ✗"]
        M_E1 --> M_SUM["Weighted Sum"]
        M_E3 --> M_SUM
        M_SUM --> M_OUT[Output]
        style M_ROUTER fill:#0d9488,color:#fff,stroke:#14b8a6
        style M_E1 fill:#14b8a6,color:#fff,stroke:#0d9488
        style M_E3 fill:#14b8a6,color:#fff,stroke:#0d9488
        style M_E2 fill:#3f3f46,color:#71717a,stroke:#52525b,stroke-dasharray: 5 5
        style M_E4 fill:#3f3f46,color:#71717a,stroke:#52525b,stroke-dasharray: 5 5
    end

    style DENSE fill:#18181b,color:#fafafa,stroke:#3f3f46
    style MOE fill:#18181b,color:#fafafa,stroke:#3f3f46
{{< /mermaid >}}

The diagram above shows the key difference: in a dense layer (left), every token uses the full FFN. In an MoE layer (right), the router selects only 2 of 4 experts (shown in teal), while the others (grayed, dashed) stay idle. The active experts' outputs are combined using the router's weights.

So if you have 8 experts and pick K=2, each token activates **25% of the FFN parameters**. The model has the total capacity of 8 experts but the per-token cost of roughly 2. This is called **sparse activation** — the defining property of MoE.

---

## The Gating Function

Given an input token \\(x\\), the MoE layer computes:

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

where:
- \\(E_i(x)\\) is the output of expert \\(i\\) — each expert is a standard FFN (linear → activation → linear)
- \\(G(x)_i\\) is the **gating weight** for expert \\(i\\)
- \\(N\\) is the total number of experts

The gating function is simple:

$$G(x) = \text{TopK}\big(\text{Softmax}(W_g \cdot x + \epsilon)\big)$$

where \\(W_g\\) is a learned weight matrix and \\(\epsilon\\) is optional noise added during training for exploration. The router multiplies the token embedding by \\(W_g\\) to score each expert, applies softmax to normalize, then keeps only the top-K probabilities (zeroing the rest). The surviving probabilities serve as weights to combine expert outputs. The router is a single linear layer — it adds almost no parameters or compute.

To make this concrete, consider a single token routed through 4 experts with K=2:

| Step | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|------|----------|----------|----------|----------|
| Raw score \\(x \cdot W_g\\) | 2.1 | 0.3 | 1.8 | −0.5 |
| After softmax | 0.41 | 0.07 | 0.30 | 0.03 |
| After Top-2 | **0.58** | 0 | **0.42** | 0 |
| Runs? | **Yes** | No | **Yes** | No |

Experts 1 and 3 scored highest and survive the top-K selection. Their softmax probabilities (0.41 and 0.30) are **renormalized** to sum to 1, becoming 0.58 and 0.42. Experts 2 and 4 are zeroed out — they contribute no computation for this token.

The final output is simply a weighted blend of the two winners:

$$y = 0.58 \cdot E_1(x) + 0.42 \cdot E_3(x)$$

---

## Why Sparse Beats Dense

Here's the core tradeoff, using Mixtral 8x7B as a concrete example:

{{< chart >}}
type: 'bar',
data: {
  labels: ['Dense 7B\n(Llama 2 7B)', 'MoE 8x7B\n(Mixtral)', 'Dense 70B\n(Llama 2 70B)'],
  datasets: [{
    label: 'Total Parameters (B)',
    data: [7, 46.7, 70],
    backgroundColor: 'rgba(20, 184, 166, 0.6)',
    borderColor: 'rgb(20, 184, 166)',
    borderWidth: 1
  }, {
    label: 'Active Parameters per Token (B)',
    data: [7, 12.9, 70],
    backgroundColor: 'rgba(251, 191, 36, 0.6)',
    borderColor: 'rgb(251, 191, 36)',
    borderWidth: 1
  }]
},
options: {
  plugins: {
    title: { display: true, text: 'Total vs Active Parameters (Billions)', color: '#a1a1aa' },
    legend: { labels: { color: '#a1a1aa' } }
  },
  scales: {
    y: { title: { display: true, text: 'Billions', color: '#a1a1aa' }, ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(161,161,170,0.1)' } },
    x: { ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(161,161,170,0.1)' } }
  }
}
{{< /chart >}}

Mixtral has 8 experts of ~7B parameters each, totaling **46.7B parameters**. But with K=2 routing, only **~12.9B parameters** are active per token. Despite using less than a fifth of Llama 2 70B's compute per token, Mixtral matches or exceeds it on most benchmarks — MMLU, HellaSwag, ARC, and code generation.

The model stores 47B parameters of knowledge but computes with only 13B per token — paying the memory cost of the large model but achieving the inference speed of a much smaller one.

---

## The Load Balancing Problem

Left unconstrained, the router will often collapse — sending most tokens to a few "popular" experts while others atrophy. This is **expert collapse**, and it wastes the capacity of unused experts.

The standard fix, introduced by Shazeer et al. (2017) and refined in the Switch Transformer, is an auxiliary **load balancing loss**:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

where:
- \\(f_i\\) = fraction of tokens actually routed to expert \\(i\\) in the batch
- \\(p_i\\) = mean router probability assigned to expert \\(i\\) across the batch
- \\(\alpha\\) = a small coefficient (typically 0.01) to avoid overpowering the main loss
- \\(N\\) = number of experts

This loss is minimized when tokens are **uniformly distributed** across experts. It nudges the router toward balance without forcing it — experts can still specialize, they just can't monopolize all the traffic.

---

## The Evolution: Four Key Papers

The MoE idea dates back to **1991**, but it took over two decades before it was successfully applied at scale in deep learning.

### Jacobs et al. (1991) — The Original

The foundational paper introduced the idea of competing expert networks with a gating function. Each expert was a small neural network, and the gate learned to partition the input space. The key insight: **competition between experts leads to specialization**. But compute was too limited for this to scale.

### Shazeer et al. (2017) — Sparsely-Gated MoE

The breakthrough paper. Shazeer et al. showed that MoE could be inserted between transformer layers and scaled to **137 billion parameters** — the largest model at the time. This paper introduced:
- Top-K gating with learned noise for exploration
- The load balancing auxiliary loss
- Proof that sparse models could match or beat dense models of similar compute budget

### Switch Transformer (Fedus et al., 2022) — Radical Simplicity

The Switch Transformer made one key simplification: **K=1**. Route each token to exactly one expert.

This appears to lose information, but the authors demonstrated it works well in practice. Different tokens within the same sequence are routed to different experts, so the model still uses multiple experts per sentence — just not per token. The simplified routing enabled scaling to **1.6 trillion parameters** with 4-7x training speedups over dense T5 models at equal compute budgets.

### Mixtral 8x7B (Jiang et al., 2024) — MoE Goes Mainstream

Mixtral proved MoE could be practical and open-source. Key design choices:
- 8 experts per layer, K=2 routing
- 46.7B total parameters, 12.9B active per token
- Each expert is a full FFN — no parameter sharing between experts
- Released with open weights, enabling community research

On benchmarks, Mixtral 8x7B matched or exceeded Llama 2 70B while being roughly **6x faster** at inference. It also outperformed GPT-3.5 on most evaluations.

---

## DeepSeek's Innovation: Shared + Routed Experts

DeepSeek-V2 (2024) introduced a meaningful architectural twist. Instead of making all experts conditionally routed, some are **always active** (shared) while others are selectively activated (routed):

- **2 shared experts** — always process every token. They capture common, domain-agnostic knowledge: grammar, basic reasoning, formatting patterns.
- **160 routed experts, top-6 selection** — conditionally activated per token. They encode specialized knowledge: domain-specific vocabulary, advanced reasoning patterns, task-specific computation.

{{< mermaid >}}
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '13px', 'primaryColor': '#14b8a6', 'primaryTextColor': '#fafafa', 'lineColor': '#5eead4', 'secondaryColor': '#27272a', 'tertiaryColor': '#3f3f46' }}}%%
graph TB
    IN[Token x after Self-Attention] --> SHARED
    IN --> ROUTER

    subgraph SHARED["<b>Always Active</b>"]
        S1["Shared Expert 1<br/><i>Grammar, syntax, formatting</i>"]
        S2["Shared Expert 2<br/><i>Basic reasoning, common knowledge</i>"]
    end

    subgraph ROUTER_BLOCK["<b>Conditionally Routed · Top-6 of 160</b>"]
        ROUTER["Router"] --> R1["Expert 12 ✓"]
        ROUTER --> R2["Expert 47 ✓"]
        ROUTER --> R3["Expert 103 ✓"]
        ROUTER --> RD["... 3 more selected"]
        ROUTER -.-> RX["154 experts idle"]
    end

    S1 --> SUM["Sum All Outputs"]
    S2 --> SUM
    R1 --> SUM
    R2 --> SUM
    R3 --> SUM
    RD --> SUM
    SUM --> OUT[Output]

    style SHARED fill:#f59e0b20,color:#fafafa,stroke:#f59e0b
    style ROUTER_BLOCK fill:#14b8a620,color:#fafafa,stroke:#14b8a6
    style S1 fill:#f59e0b,color:#18181b,stroke:#d97706
    style S2 fill:#f59e0b,color:#18181b,stroke:#d97706
    style ROUTER fill:#0d9488,color:#fff,stroke:#14b8a6
    style R1 fill:#14b8a6,color:#fff,stroke:#0d9488
    style R2 fill:#14b8a6,color:#fff,stroke:#0d9488
    style R3 fill:#14b8a6,color:#fff,stroke:#0d9488
    style RD fill:#14b8a6,color:#fff,stroke:#0d9488
    style RX fill:#3f3f46,color:#71717a,stroke:#52525b,stroke-dasharray: 5 5
    style IN fill:#27272a,color:#fafafa,stroke:#3f3f46
    style OUT fill:#27272a,color:#fafafa,stroke:#3f3f46
    style SUM fill:#27272a,color:#fafafa,stroke:#5eead4
{{< /mermaid >}}

The total model has **236B parameters** but activates only **21B per token**. The shared experts (amber) always run, capturing domain-agnostic computation. The routed experts (teal) are conditionally selected for specialized tasks. This design improved performance over pure MoE at the same compute budget. The intuition: some knowledge genuinely is universal (you always need grammar), so forcing the router to "rediscover" it for every token is wasteful.

---

## What Each Expert Actually Learns

One of the most interesting findings from the Mixtral paper is that experts **naturally specialize** without being told to. The authors analyzed which experts fired most frequently for different types of text:

| Domain | Most Active Experts | Observation |
|--------|-------------------|-------------|
| Code (Python, C++) | Experts 0, 3 | Consistent across programming languages |
| Mathematics | Experts 2, 5 | Overlaps with formal reasoning |
| English prose | Experts 1, 4 | Distinct from code experts |
| Multilingual text | Experts 4, 7 | Partially overlaps with English |
| Structured data (JSON, XML) | Experts 0, 6 | Overlaps with code |

This specialization is **emergent**. The training objective is just next-token prediction — there's no auxiliary signal telling Expert 3 to handle code. The router learns the partitioning purely from the data.

Importantly, the specialization is not absolute. Most tokens activate 2 experts, and those pairs vary. It's more like soft preferences than hard assignments.

---

## The Practical Tradeoffs

{{< accordion >}}

{{< accordionItem title="When should you use MoE?" open=true >}}
**MoE makes sense when:**
- You need a very capable model but have a limited inference compute budget
- Your workload is diverse — code, language, math, reasoning across many domains
- You can afford the memory for all expert parameters (even inactive ones must be loaded)

**A dense model is better when:**
- Your task is narrow — a domain-specific fine-tuned dense model will be smaller and cheaper
- Memory, not compute, is your bottleneck — MoE uses more memory than a dense model of equivalent inference FLOPs
- You need fully deterministic behavior — routing introduces minor non-determinism
{{< /accordionItem >}}

{{< accordionItem title="MoE vs Dense: concrete numbers" >}}
| Metric | Llama 2 70B (Dense) | Mixtral 8x7B (MoE) |
|--------|---------------------|---------------------|
| Total parameters | 70B | 46.7B |
| Active per token | 70B | 12.9B |
| Inference cost (relative) | 1x | ~0.18x |
| Memory required (FP16) | ~140 GB | ~94 GB |
| MMLU (5-shot) | 69.8% | 70.6% |
| HellaSwag | 85.7% | 86.7% |
| HumanEval (code) | 29.9% | 40.2% |

*Benchmark numbers from the Mixtral technical report (Jiang et al., 2024).*
{{< /accordionItem >}}

{{< accordionItem title="Open problems" >}}
1. **Expert collapse** — Some experts receive disproportionately few tokens and never fully train. Load balancing helps but doesn't eliminate it.
2. **Routing instability** — The router can oscillate during training, assigning the same token to different experts across steps.
3. **Memory overhead** — All N experts must be resident in memory even though only K are active. This limits deployment on consumer hardware.
4. **Fine-tuning fragility** — Fine-tuning MoE models can destabilize the learned expert specialization, especially with small datasets.
5. **Interpretability** — We observe emergent specialization but don't fully control or understand it. Expert roles can shift during training.
{{< /accordionItem >}}

{{< /accordion >}}

---

## Conclusion

Mixture of Experts decouples model capacity from per-token compute cost. The core mechanism — a learned router that selects a sparse subset of expert FFNs per token — is simple enough to describe in a single equation, yet powerful enough to underpin the largest language models in production today.

The trajectory from Jacobs et al. (1991) through Shazeer's sparsely-gated MoE, the Switch Transformer's K=1 simplification, Mixtral's open-source demonstration, and DeepSeek's shared-routed architecture reflects a consistent theme: sparser routing, fewer assumptions, and larger scale. Each generation validated that expert specialization emerges naturally from the training signal without explicit supervision.

Open challenges remain — expert collapse, routing instability, memory overhead, and fine-tuning fragility all limit practical deployment. But the fundamental insight is now well-established: conditional computation through sparse expert selection is one of the most effective tools for scaling language model capabilities without proportionally scaling inference cost.

---

## References

1. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). *Adaptive Mixtures of Local Experts*. Neural Computation, 3(1), 79-87.
2. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
3. Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR, 23(120), 1-39.
4. Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). *Mixtral of Experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
5. DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434).
