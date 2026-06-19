# 05 — Dense vs MoE: The Architecture Beyond

*Part 5 of 5 — From Neuron to Agent*
*Series index: [README.md](./README.md)*

---

## Overview

This document covers the architectural difference between dense and Mixture of Experts (MoE) models, traced through Project 11 (8 experts, top-2 routing, built from scratch) and Gemma 4 26B A4B (26B total parameters, 4B active per token, production MoE).

The build series was entirely dense — every parameter active on every token, every forward pass. Gemma 4 26B changes that. Understanding why requires understanding what MoE is, what it costs, and what it saves.

---

## Dense architecture — what the build series built

Every model in Projects 1–11 was dense. In a dense Transformer block the feed-forward network (FFN) activates all its parameters for every input token.

From `mini_llm.py` (Project 7):

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden_dim, dropout_rate):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(embedding_dim, ffn_hidden_dim)
        self.fully_connected_layer_2 = nn.Linear(ffn_hidden_dim, embedding_dim)
        self.activation_function     = nn.GELU()
        self.dropout                 = nn.Dropout(dropout_rate)

    def forward(self, input_tensor):
        hidden_states  = self.fully_connected_layer_1(input_tensor)
        activated      = self.activation_function(hidden_states)
        dropped        = self.dropout(activated)
        output_tensor  = self.fully_connected_layer_2(dropped)
        return output_tensor
```

For every token, every forward pass: all parameters in `fully_connected_layer_1` and `fully_connected_layer_2` participate in the computation. No selection. No routing. Total parameter count equals active parameter count.

In Project 7:

- Total parameters: 159,558
- Active parameters per token: 159,558 (100%)

---

## Mixture of Experts — what Project 11 built

Project 11 replaced the single FFN in each Transformer block with a set of expert FFNs plus a router. The router selects which experts handle each token. The other experts do not participate in that forward pass.

```python
class MixtureOfExpertsLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden_dim, number_of_experts, top_k_experts):
        super().__init__()
        self.number_of_experts = number_of_experts
        self.top_k_experts     = top_k_experts

        # One FFN per expert
        self.expert_networks = nn.ModuleList([
            FeedForwardNetwork(embedding_dim, ffn_hidden_dim)
            for _ in range(number_of_experts)
        ])

        # Router: maps each token's embedding to a score per expert
        self.router = nn.Linear(embedding_dim, number_of_experts, bias=False)

    def forward(self, input_tensor):
        batch_size, sequence_length, embedding_dim = input_tensor.shape
        tokens_flat = input_tensor.view(-1, embedding_dim)

        # Router produces logits over experts for each token
        router_logits    = self.router(tokens_flat)
        router_probs     = torch.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k_experts, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute weighted sum of selected expert outputs
        output_tensor = torch.zeros_like(tokens_flat)

        for expert_index in range(self.number_of_experts):
            expert_mask   = (top_k_indices == expert_index).any(dim=-1)
            expert_tokens = tokens_flat[expert_mask]

            if expert_tokens.shape[0] == 0:
                continue

            expert_output = self.expert_networks[expert_index](expert_tokens)

            expert_weights = top_k_probs[expert_mask][
                top_k_indices[expert_mask] == expert_index
            ].unsqueeze(-1)

            output_tensor[expert_mask] += expert_weights * expert_output

        return output_tensor.view(batch_size, sequence_length, embedding_dim)
```

Project 11 configuration:

- Total experts: 8
- Top-k experts activated per token: 2
- Active experts per forward pass: 2 out of 8 (25%)

The router is a small linear layer. Its job is to score each expert for each token and return the top-k indices. The experts themselves are standard FFNs — the same structure as in Projects 1–7, just replicated N times.

---

## Auxiliary load balancing loss

A MoE model without load balancing collapses. The router learns to always send tokens to the same 1–2 experts, leaving the rest unused. This is called expert collapse and it wastes the capacity you built MoE to gain.

Project 11 added an auxiliary loss term to penalise uneven expert utilisation:

```python
def compute_load_balancing_loss(
    router_probs,
    top_k_indices,
    number_of_experts,
    load_balancing_loss_weight
):
    # Fraction of tokens routed to each expert
    expert_mask          = F.one_hot(top_k_indices, num_classes=number_of_experts).float()
    tokens_per_expert    = expert_mask.mean(dim=[0, 1])

    # Mean routing probability for each expert
    mean_router_probs    = router_probs.mean(dim=[0, 1])

    # Auxiliary loss: penalise deviation from uniform distribution
    load_balancing_loss  = number_of_experts * (tokens_per_expert * mean_router_probs).sum()

    return load_balancing_loss_weight * load_balancing_loss


# Training loop
total_loss = cross_entropy_loss + compute_load_balancing_loss(
    router_probs=router_probs,
    top_k_indices=top_k_indices,
    number_of_experts=8,
    load_balancing_loss_weight=0.01
)
```

The auxiliary loss encourages the router to distribute tokens roughly evenly across experts. Without it, training converges but the model is effectively dense — only 1–2 experts ever activate.

Production MoE models including Gemma 4 26B use the same mechanism at scale.

---

## Gemma 4 26B A4B — production MoE

Gemma 4 26B A4B is the production MoE variant in the Gemma 4 family. The "A4B" designation means 4 billion **active** parameters per token.

| Property | Value | Source |
| --- | --- | --- |
| Total parameters | 26B | Google AI docs |
| Active parameters per token | 4B | Google AI docs |
| All weights loaded into memory | Yes | Google AI docs |
| Expert count per layer | Not disclosed | Google has not published this |
| Architecture | MoE replacing FFN layers | Gemma 4 model card |

Key constraint from the official docs: **While it only activates 4 billion parameters per token during generation, all 26 billion parameters must be loaded into memory to maintain fast routing and inference speeds. This is why its baseline memory requirement is much closer to a dense 26B model than a 4B model.**

This is the same constraint visible in Project 11 — all expert networks are instantiated and loaded at `__init__` time, even though only 2 out of 8 activate per token. Memory cost is total parameters. Compute cost is active parameters.

---

## Dense vs MoE: direct comparison

| Property | Dense | MoE |
| --- | --- | --- |
| Parameters that activate per token | All | Top-k out of N |
| Memory requirement | Total params | Total params (all experts must be loaded) |
| FLOPs per token | High (proportional to total params) | Low (proportional to active params) |
| Knowledge capacity | Bounded by total params | Higher — more total params at same FLOPs |
| Expert collapse risk | None | Yes — requires load balancing loss |
| Router overhead | None | Small linear layer per MoE layer |
| Implementation complexity | Low | Higher — routing, masking, load balancing |

The core insight: **active parameters determine compute cost per token. Total parameters determine knowledge capacity.** MoE decouples these two quantities. A dense model with the same compute budget as Gemma 4 26B A4B would have 4B total parameters, not 26B. The MoE model stores 6.5x more knowledge at the same inference cost.

---

## Build series to production: direct mapping

| Project 11 (MoE from scratch) | Gemma 4 26B A4B |
| --- | --- |
| 8 experts per MoE layer | Not disclosed per layer |
| Top-2 routing | Activates 4B out of 26B per token |
| Router: `nn.Linear(embedding_dim, 8)` | Equivalent lightweight router |
| Auxiliary load balancing loss (weight=0.01) | Equivalent mechanism at scale |
| All 8 experts loaded into memory | All 26B parameters loaded into memory |
| Active: 2/8 experts (25%) | Active: ~15% of total parameters |
| Single GPU, toy corpus | Distributed across data center hardware |

The mechanism is identical. The scale is not.

---

## Why the Gemma 4 family includes both Dense and MoE

Gemma 4 ships four variants:

| Variant | Architecture | Total params | Active params | Use case |
| --- | --- | --- | --- | --- |
| E2B | Dense (MobileNet-style) | ~2B effective | ~2B | Mobile, edge |
| E4B | Dense (MobileNet-style) | ~4B effective | ~4B | Mobile, laptop |
| 26B A4B | MoE | 26B | 4B | High-throughput server |
| 31B | Dense | 31B | 31B | Maximum quality, server |

MoE is not universally superior. Dense models are simpler to serve, have no router overhead, and do not require all experts to be loaded simultaneously into a single accelerator's memory. The 31B dense variant reaches higher benchmark scores than the 26B MoE variant precisely because every parameter contributes to every token.

The choice between them is an engineering tradeoff, not a quality ranking.

---

## Memory footprint in practice

| Model | Architecture | Memory (full precision) | Memory (4-bit NF4) |
| --- | --- | --- | --- |
| Project 7 mini LLM | Dense | negligible | N/A |
| Gemma 4 12B (this project) | Dense | ~24 GB | ~7.5 GB |
| Gemma 4 26B A4B | MoE | ~52 GB | ~13 GB |
| Gemma 4 31B | Dense | ~62 GB | ~16 GB |

Gemma 4 26B requires approximately 52 GB in full precision — more than two T4 GPUs combined. Running it on Kaggle requires at minimum 4-bit quantization and a high-RAM instance.

---

## References

- Gemma 4 model overview: <https://ai.google.dev/gemma/docs/core>
- Gemma 4 model card: <https://ai.google.dev/gemma/docs/core/model_card_4>
- Mixtral MoE paper (Jiang et al., 2024): <https://arxiv.org/abs/2401.04088>
- Switch Transformers (Fedus et al., 2022): <https://arxiv.org/abs/2101.03961>
- Project 11 source: [../11-Mixture_of_Expert/](../11-Mixture_of_Expert/)
- Gemma 4 26B HuggingFace: <https://huggingface.co/google/gemma-4-27b-it>

---

*[← 04 — The ReAct Loop](./04-react-loop-from-scratch.md) | [Series Index](./README.md) | No next article →*
