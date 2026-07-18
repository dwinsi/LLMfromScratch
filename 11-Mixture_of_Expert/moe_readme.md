# 11: Mixture of Experts

Every project in this series has had one thing in common: every neuron in the network participates in processing every token. When the model reads the word "cloudy", every weight in every layer is involved in computing the output. When it reads "rain", same thing. Every token gets the full network.

This is called a **dense** model. All parameters are active at all times.

Mixture of Experts (MoE) breaks this assumption. Instead of one large feed-forward network, each Transformer block contains several smaller independent networks called **experts**. Each token is processed by only a small subset of them, typically 2 out of 8. The rest sit idle for that token.

This means the model can have far more total parameters than a dense model of the same compute cost. More parameters means more capacity to learn patterns. Same compute means training and inference stay affordable.

This is the architecture behind Gemma 4 (128 experts, top-2 routing) and almost certainly GPT-4.

---

## The core idea: specialisation through routing

Think about how a company works. A large company does not have every employee work on every problem. It has specialists: engineers for the product, accountants for the finances, lawyers for contracts. When a problem arrives, it gets routed to the people best suited to handle it.

MoE works the same way. Each expert network learns to specialise in certain kinds of tokens or patterns. The router learns to recognise which tokens belong to which experts and sends them accordingly.

After training:

- Some experts might specialise in common words ("the", "and", "is")
- Others might specialise in weather-related patterns ("cloudy", "rain", "storm")
- Others might specialise in action words ("bring", "carry", "stop")

This specialisation emerges automatically. Nobody programs it. The router and the experts learn their roles together through training.

---

## What changes from Project 10

Projects 9 and 10 built a LLaMA-style Transformer with RMSNorm, RoPE, SwiGLU, and Grouped Query Attention. This project keeps all of that and replaces exactly one thing:

```text
Project 10:
  TransformerBlock:
    RMSNorm
    Grouped Query Attention (RoPE)
    Residual
    RMSNorm
    SwiGLU FFN  <-- one dense feed-forward network
    Residual

Project 11:
  TransformerBlock:
    RMSNorm
    Grouped Query Attention (RoPE)
    Residual
    RMSNorm
    Mixture of Experts  <-- 8 expert networks + 1 router
    Residual
```

The attention sublayer, normalisation, residual connections, BPE tokenisation, batching, and training setup are all unchanged.

---

## The three new components

### 1. Expert networks

Each expert is a complete SwiGLU feed-forward network, identical to the one used in Projects 9 and 10. There are 8 of them per Transformer block, each with its own independent weight matrices:

```python
class ExpertNetwork(nn.Module):
    def __init__(self, embedding_dim, feedforward_hidden_dim):
        super(ExpertNetwork, self).__init__()
        self.gate_projection     = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.value_projection    = nn.Linear(embedding_dim, feedforward_hidden_dim, bias=False)
        self.compress_projection = nn.Linear(feedforward_hidden_dim, embedding_dim, bias=False)

    def forward(self, x):
        gate   = F.silu(self.gate_projection(x))
        value  = self.value_projection(x)
        hidden = gate * value
        return self.compress_projection(hidden)
```

Each expert sees only the tokens that were routed to it. It has no knowledge of which tokens went to other experts. Over many training steps, each expert receives a different mix of tokens and its weights drift to become good at the patterns it sees most often.

Eight experts are stored in an `nn.ModuleList`:

```python
self.experts = nn.ModuleList([
    ExpertNetwork(embedding_dim, feedforward_hidden_dim)
    for _ in range(number_of_experts)
])
```

### 2. The router

The router is a single small linear layer that takes a token representation and produces a score for each of the 8 experts:

```python
self.router = nn.Linear(embedding_dim, number_of_experts, bias=False)
```

For a token with a 64-dimensional embedding, the router produces 8 numbers, one per expert. These are passed through softmax to become probabilities that sum to 1:

```text
Input token representation: [0.3, -0.1, 0.8, ..., 0.2]   (64 numbers)

Router output (logits):   [1.2,  0.4, -0.3,  0.8,  0.1, -0.7,  0.9,  0.2]
After softmax (probs):    [0.28, 0.12,  0.06, 0.18, 0.09,  0.04, 0.17, 0.09]
                             ^                  ^                  ^
                          Expert 0           Expert 3           Expert 6
                         gets 28%           gets 18%           gets 17%
```

These probabilities answer the question: "how relevant is each expert for this particular token?"

### 3. Top-K selection

After computing router probabilities, only the top 2 experts (by probability) are actually used. The other 6 are skipped entirely for this token:

```python
top_k_probs, top_k_indices = torch.topk(
    router_probs, self.number_of_active_experts, dim=-1
)
```

For the example above, the top-2 would be Expert 0 (28%) and Expert 3 (18%). The selected probabilities are then renormalised so they sum to 1:

```text
Selected:    Expert 0: 0.28,  Expert 3: 0.18
Sum:         0.46
Renormalised: Expert 0: 0.28/0.46 = 0.61,  Expert 3: 0.18/0.46 = 0.39
```

The renormalised weights are used to blend the two expert outputs. If Expert 0 says "next token should be X" with strength 0.61 and Expert 3 says "next token should be Y" with strength 0.39, the final output is a weighted mix of both.

---

## The full forward pass through MixtureOfExperts

```python
def forward(self, token_representations):
    # token_representations: (batch, seq_len, embedding_dim)

    # Flatten all tokens into a single list for routing
    num_tokens  = batch_size * seq_len
    flat_tokens = token_representations.view(num_tokens, embed_dim)

    # Step 1: router scores every expert for every token
    router_logits = self.router(flat_tokens)               # (num_tokens, 8)
    router_probs  = torch.softmax(router_logits, dim=-1)   # (num_tokens, 8)

    # Step 2: select top-2 experts per token
    top_k_probs, top_k_indices = torch.topk(router_probs, 2, dim=-1)
    top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Step 3: process tokens through their selected experts
    moe_output = torch.zeros_like(flat_tokens)

    for expert_index in range(8):
        # Which tokens chose this expert?
        token_mask = (top_k_indices == expert_index).any(dim=-1)

        if not token_mask.any():
            continue  # nobody chose this expert, skip it

        # Run only the selected tokens through this expert
        expert_input  = flat_tokens[token_mask]
        expert_output = self.experts[expert_index](expert_input)

        # Find this expert's weight for each token that selected it
        expert_position_mask = (top_k_indices[token_mask] == expert_index)
        routing_weight = (top_k_weights[token_mask] * expert_position_mask.float()).sum(dim=-1, keepdim=True)

        # Add weighted output back to the result
        moe_output[token_mask] += routing_weight * expert_output

    # Reshape back to (batch, seq_len, embed_dim)
    moe_output = moe_output.view(batch_size, seq_len, embed_dim)
```

Notice the `if not token_mask.any(): continue` line. If an expert was not selected by any token in this batch, it is skipped entirely. Its forward pass does not run. Its parameters do not receive gradients. This is the computational saving: most experts are idle for most tokens in each batch.

---

## The router collapse problem

Left to its own devices, the router will learn to always route tokens to whichever expert it found most useful early in training. If Expert 0 happens to do slightly better than the others in the first few batches, the router starts sending more tokens there. Expert 0 receives more gradient updates, gets better, and the router sends even more tokens to it. Eventually everything goes to one or two experts and the rest are never used.

This is called **router collapse**. It defeats the entire purpose of MoE: you end up with a dense model (all compute going to one expert) but with the memory footprint of eight experts.

### The load balancing loss

To prevent router collapse, a second loss term is added that penalises uneven expert utilisation. The idea from the Switch Transformer paper (Google, 2021) is to measure two quantities across the batch:

- **fraction of tokens routed to each expert**: what percentage of all token-routing decisions chose each expert?
- **mean router probability assigned to each expert**: on average, how much probability mass does the router place on each expert?

If the routing is perfectly balanced, both of these would be `1 / num_experts = 1/8 = 0.125` for every expert.

The auxiliary loss is:

```text
L_aux = num_experts * sum(fraction_i * mean_prob_i)
```

When the routing is perfectly uniform: `8 * 8 * (0.125 * 0.125) = 8 * 0.125 = 1.0`

When routing collapses to one expert: that expert has fraction=1.0 and mean_prob near 1.0. The loss becomes `8 * 1.0 * 1.0 = 8.0`, which is much higher.

The loss pushes the router toward uniform distribution over experts, preventing collapse.

In code:

```python
# fraction of times each expert was chosen
expert_selection_counts = torch.zeros(self.number_of_experts, ...)
for k in range(self.number_of_active_experts):
    expert_selection_counts.scatter_add_(0, top_k_indices[:, k], ones)

fraction_per_expert     = expert_selection_counts / (num_tokens * self.number_of_active_experts)
mean_prob_per_expert    = router_probs.mean(dim=0)

load_balancing_loss = self.number_of_experts * torch.sum(
    fraction_per_expert * mean_prob_per_expert
)
```

### How the two losses are combined

The training loop uses two loss terms: the main cross-entropy loss (how well the model predicts the next token) and the auxiliary load balancing loss (how evenly the experts are used):

```python
cross_entropy_loss = loss_function(output_scores, batch_targets)
combined_loss      = cross_entropy_loss + auxiliary_loss_weight * load_balancing_loss

combined_loss.backward()
```

The `auxiliary_loss_weight` is a small number (typically 0.01 to 0.1). It is small enough that the primary objective remains predicting the next token correctly, but large enough to prevent router collapse.

The model also returns the load balancing loss separately so it can be logged:

```python
output_scores, load_balancing_loss = model(batch_sequences)
```

The training loop prints both values so you can monitor whether the experts are being used evenly.

---

## Parameter count: total vs active

This is where MoE becomes interesting at scale.

With 8 experts and top-2 routing, only 2 of the 8 expert networks are used per token. That means only 25% of the total expert parameters are active per forward pass.

```text
Project 10 (dense):
  1 SwiGLU FFN per block
  Total = Active for every token

Project 11 (MoE):
  8 SwiGLU FFNs per block (8x the expert parameters)
  Active per token: 2 of 8 (25%)
  Total parameters: ~8x more in the FFN layers
  Compute per token: ~2x more than dense (2 experts instead of 1)
```

In practice this means a MoE model can have 4 to 8 times the parameters of a dense model at 2 times the compute cost per token. You get more capacity (better at fitting patterns) without a proportional increase in training or inference time.

At the scale of real models:

```text
Dense model:         70 billion parameters
  Every token uses all 70B parameters

MoE equivalent:     400 billion total, 70 billion active
  Every token uses 70B parameters (same compute)
  But 400B worth of knowledge can be stored across experts
```

This is why Gemma 4 uses 128 experts with top-2 routing. The model stores an enormous amount of specialised knowledge across 128 expert networks, but inference cost stays similar to a much smaller dense model.

---

## The router visualisation

After training, the script generates two plots showing how the router behaves:

**Router heatmap**: a grid with tokens on the vertical axis and experts on the horizontal axis. Each cell shows the router probability for that token-expert pair. Bright cells mean high probability (the router strongly considers that expert for that token). Dark cells mean low probability.

**Expert utilisation bar chart**: how many times each expert was selected as the top-1 choice across a set of seed phrases. The dashed red line shows what perfect uniform utilisation would look like. If the load balancing loss is working, the bars should be roughly equal height.

These visualisations are generated using a **forward hook**: a callback function registered on the MoE layer that runs automatically every time that layer's forward method is called, capturing the router probabilities without modifying the class itself:

```python
def _router_hook(module, inputs, outputs):
    flat_tokens   = inputs[0].view(-1, inputs[0].shape[-1])
    router_logits = module.router(flat_tokens)
    router_probs  = torch.softmax(router_logits, dim=-1)
    _captured_router_probs.append(router_probs.detach().cpu())

hook_handle = first_moe_block.register_forward_hook(_router_hook)
# ... run the model ...
hook_handle.remove()
```

`register_forward_hook` is a PyTorch utility that runs a function after any module's forward pass completes, receiving the inputs and outputs. `hook_handle.remove()` unregisters it afterwards so it does not keep running.

---

## The training loop: two losses to watch

The training output now shows three numbers per checkpoint:

```text
Epoch     0  train: 5.5142  val: 5.5381  lb_loss: 1.0012  lr: 0.001000
Epoch   400  train: 2.1847  val: 2.3105  lb_loss: 0.9991  lr: 0.000905
Epoch   800  train: 1.4123  val: 1.5892  lb_loss: 0.9977  lr: 0.000655
Epoch  1200  train: 1.1045  val: 1.2891  lb_loss: 0.9983  lr: 0.000345
Epoch  1600  train: 0.9234  val: 1.1203  lb_loss: 0.9971  lr: 0.000095
Epoch  2000  train: 0.8891  val: 1.0987  lb_loss: 0.9968  lr: 0.000000
```

`train` and `val` are the cross-entropy losses (lower is better, 5.55 random baseline for 256 tokens).

`lb_loss` is the load balancing loss. A value near 1.0 means the experts are being used roughly uniformly. Values much higher than 1.0 indicate router collapse beginning. Values much lower than 1.0 would also be unusual.

---

## Why MoE matters at scale

At the scale of our weather corpus, MoE does not produce dramatically better results than a dense model. The corpus is too small and simple for specialisation to emerge clearly. What this project demonstrates is the mechanism: a router that learns to direct tokens, experts that can in principle specialise, and the load balancing trick that keeps all experts in use.

At real scale, the benefits are substantial:

**More capacity per compute budget.** A 400B parameter MoE model at 70B active parameters fits more knowledge than a 70B dense model, with similar inference cost.

**Specialisation across domains.** In a large corpus containing English text, Python code, mathematical notation, and multiple foreign languages, different experts genuinely specialise in different content types. This has been observed empirically in models like Switch Transformer and Mixtral.

**Natural scaling.** Adding more experts increases model capacity without changing the per-token compute cost. Doubling from 8 to 16 experts roughly doubles total capacity while adding only a small fraction to the compute cost per token.

---

## Architecture comparison across the series

```text
Project    Key change                            Status
-----------------------------------------------------------------
01         Single neuron                         done
02         Multi-layer network + backprop         done
03         PyTorch: nn.Module, autograd           done
04         Word-level RNN                         done
05         RNN with attention (Q/K/V)             done
06         Transformer block                      done
07         Mini LLM (4 blocks, DataLoader)        done
08         BPE tokenisation                       done
09         LLaMA-style (RMSNorm, RoPE, SwiGLU)   done
10         Grouped Query Attention (GQA)          done
11         Mixture of Experts (this project)      done
```

---

## Running

```bash
pip install torch tokenizers matplotlib
python mini_llm_moe.py
```

The script trains a BPE tokeniser, builds training sequences, trains the MoE model with combined loss, generates text samples, saves loss curves to `loss_curve_moe.png`, and saves router visualisation plots to `router_heatmap.png`.

## Files

```text
mini_llm_moe.py          full training script with MoE
weather_corpus_v2.txt    shared corpus (same as Projects 8, 9, 10)
config.json              hyperparameters including number_of_experts and auxiliary_loss_weight
loss_curve_moe.png       cross-entropy and load balancing loss over training (generated on run)
router_heatmap.png       router probabilities and expert utilisation (generated on run)
```
