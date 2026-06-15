"""
Project 11: Mixture of Experts (MoE)

Replaces the dense SwiGLU feed forward network with a Mixture of Experts layer.
8 expert networks per layer, top-2 activated per token.

This is the architecture used in Gemma 4 (128 experts, top-2) and likely GPT-4.
More total capacity, same compute per token.

Changes from Project 10:
  SwiGLU FFN (single dense network) -> MoE (8 experts, top-2 routing)
  Router network added per block
  Load balancing auxiliary loss added to training

What stays the same:
  RMSNorm, RoPE from Project 9
  Grouped Query Attention from Project 10
  BPE tokenisation from Project 8
  Four Transformer blocks
  Batching, cosine annealing, gradient clipping

Key concepts:
  Expert networks:     8 independent SwiGLU FFNs per layer
  Router:              linear layer that scores each expert for each token
  Top-K routing:       each token activates exactly 2 experts
  Load balancing loss: prevents router collapse (all tokens to same expert)
  Active ratio:        25% of expert parameters used per token (2 of 8)

Install requirements:
  pip install torch tokenizers matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

torch.manual_seed(42)

# ---- Device setup ----
try:
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except Exception:
    device = torch.device('cpu')

print(f"Using device: {device}")


# ---- Load corpus and train BPE tokeniser ----

corpus_file_path    = 'weather_corpus_v2.txt'
tokenizer_save_path = 'weather_bpe_tokenizer.json'

with open(corpus_file_path, 'r') as f:
    corpus_text = f.read().lower()

bpe_tokenizer               = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = ByteLevel()
bpe_tokenizer.decoder       = ByteLevelDecoder()

bpe_trainer = BpeTrainer(
    vocab_size=256,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    min_frequency=1
)
bpe_tokenizer.train(files=[corpus_file_path], trainer=bpe_trainer)
bpe_tokenizer.save(tokenizer_save_path)

vocabulary_size = bpe_tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocabulary_size}")


# ---- Build training sequences ----

batch_size         = 32
sequence_length    = 8
training_sequences = []
training_targets   = []

all_token_ids = bpe_tokenizer.encode(corpus_text).ids

for i in range(len(all_token_ids) - sequence_length):
    training_sequences.append(all_token_ids[i : i + sequence_length])
    training_targets.append(all_token_ids[i + sequence_length])

sequences_tensor = torch.tensor(training_sequences)
targets_tensor   = torch.tensor(training_targets)

training_dataset = TensorDataset(sequences_tensor, targets_tensor)
training_loader  = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

print(f"Training sequences: {len(training_sequences)}")


# ---- RMSNorm (unchanged from Project 9) ----

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, epsilon=1e-6):
        super(RMSNorm, self).__init__()
        self.epsilon       = epsilon
        self.learned_scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms        = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)
        normalised = x / rms
        return self.learned_scale * normalised


# ---- RoPE (unchanged from Project 9) ----

def compute_rope_frequencies(head_dim, max_seq_len, base=10000, device='cpu'):
    dimension_pair_indices = torch.arange(0, head_dim, 2, device=device).float()
    frequencies            = 1.0 / (base ** (dimension_pair_indices / head_dim))
    positions              = torch.arange(max_seq_len, device=device).float()
    angles                 = torch.outer(positions, frequencies)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(query_or_key, cos_table, sin_table):
    seq_len  = query_or_key.shape[1]
    cos_vals = cos_table[:seq_len].unsqueeze(0).unsqueeze(2)
    sin_vals = sin_table[:seq_len].unsqueeze(0).unsqueeze(2)
    x_even   = query_or_key[..., 0::2]
    x_odd    = query_or_key[..., 1::2]
    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd  = x_even * sin_vals + x_odd * cos_vals
    x_rotated      = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    return x_rotated.flatten(-2)


# ============================================================
# NEW: MIXTURE OF EXPERTS FEED FORWARD
# ============================================================

class ExpertNetwork(nn.Module):
    """
    One expert: a single SwiGLU feed forward network.
    Identical architecture to Project 9/10 FFN.
    Each expert learns to specialise in different token patterns through training.
    """

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


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer replacing the dense SwiGLU FFN.

    For each token:
      1. Router scores all experts
      2. Top-K experts are selected
      3. Selected experts process the token independently
      4. Outputs are combined as a weighted sum

    Also computes a load balancing auxiliary loss to prevent
    router collapse (all tokens going to the same expert).

    With 8 experts and top-2 routing:
      Total parameters: 8 × FFN_params + router_params
      Active per token: 2 × FFN_params  (25% of total expert capacity)
    """

    def __init__(self, embedding_dim, feedforward_hidden_dim,
                 number_of_experts, number_of_active_experts):
        super(MixtureOfExperts, self).__init__()

        self.number_of_experts        = number_of_experts
        self.number_of_active_experts = number_of_active_experts

        # 8 independent expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(embedding_dim, feedforward_hidden_dim)
            for _ in range(number_of_experts)
        ])

        # Router: small linear layer that scores each expert for each token
        # No bias, no activation - just a linear projection to expert scores
        self.router = nn.Linear(embedding_dim, number_of_experts, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, token_representations):
        """
        token_representations: (batch, seq_len, embedding_dim)
        Returns: (output, auxiliary_load_balancing_loss)
        """
        batch_size = token_representations.shape[0]
        seq_len    = token_representations.shape[1]
        embed_dim  = token_representations.shape[2]

        # Flatten tokens for routing: (batch * seq_len, embed_dim)
        num_tokens  = batch_size * seq_len
        flat_tokens = token_representations.view(num_tokens, embed_dim)

        # ---- Step 1: Router scores all experts for each token ----
        router_logits = self.router(flat_tokens)              # (num_tokens, num_experts)
        router_probs  = torch.softmax(router_logits, dim=-1)  # (num_tokens, num_experts)

        # ---- Step 2: Select top-K experts per token ----
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.number_of_active_experts, dim=-1
        )   # both shape: (num_tokens, number_of_active_experts)

        # Renormalise selected probabilities to sum to 1
        # This ensures the weighted combination is a proper weighted average
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # ---- Step 3: Each token processed by its selected experts ----
        moe_output = torch.zeros_like(flat_tokens)

        for expert_index in range(self.number_of_experts):

            # Find which tokens selected this expert and at which position
            # token_mask: (num_tokens,) True where this expert was selected
            token_mask = (top_k_indices == expert_index).any(dim=-1)

            if not token_mask.any():
                continue  # this expert was not selected by any token this batch

            # Get the tokens assigned to this expert
            expert_input = flat_tokens[token_mask]   # (selected_tokens, embed_dim)

            # Run the expert
            expert_output = self.experts[expert_index](expert_input)
            expert_output = self.dropout(expert_output)

            # Get the routing weight for this expert for each selected token
            # Find the position (0 or 1) where this expert appears in top_k_indices
            expert_position_mask = (top_k_indices[token_mask] == expert_index)
            routing_weight       = (top_k_weights[token_mask] * expert_position_mask.float()).sum(dim=-1, keepdim=True)

            # Weighted add to output
            moe_output[token_mask] += routing_weight * expert_output

        # Reshape back to (batch, seq_len, embed_dim)
        moe_output = moe_output.view(batch_size, seq_len, embed_dim)

        # ---- Step 4: Load balancing auxiliary loss ----
        # Prevents router collapse where all tokens go to the same experts
        #
        # Intuition: we want two things to be uniform across experts
        #   1. The fraction of tokens routed to each expert
        #   2. The average router probability assigned to each expert
        #
        # The loss is high when popular experts also get high router probabilities
        # This pushes the router toward uniform distribution
        #
        # Formula from Switch Transformer paper:
        #   L_aux = num_experts * sum(fraction_i * mean_prob_i)
        #   where fraction_i = tokens routed to expert i / total tokens
        #         mean_prob_i = mean router probability for expert i across all tokens

        # Fraction of tokens routed to each expert
        # Create a one-hot style indicator for which expert each token chose
        # Sum across top-k selections then normalise
        expert_selection_counts = torch.zeros(
            self.number_of_experts, device=flat_tokens.device
        )
        for k in range(self.number_of_active_experts):
            expert_selection_counts.scatter_add_(
                0,
                top_k_indices[:, k],
                torch.ones(num_tokens, device=flat_tokens.device)
            )

        fraction_of_tokens_per_expert = expert_selection_counts / (num_tokens * self.number_of_active_experts)
        mean_router_probability_per_expert = router_probs.mean(dim=0)

        load_balancing_loss = self.number_of_experts * torch.sum(
            fraction_of_tokens_per_expert * mean_router_probability_per_expert
        )

        return moe_output, load_balancing_loss


# ============================================================
# UPDATED TRANSFORMER BLOCK WITH MoE
# ============================================================

class TransformerBlock(nn.Module):
    """
    LLaMA-style Transformer block with Mixture of Experts FFN.

    Change from Project 10:
      SwiGLU FFN (dense) -> MixtureOfExperts (8 experts, top-2)

    The attention sublayer (GQA + RoPE) is unchanged.
    """

    def __init__(self, embedding_dim, number_of_query_heads, number_of_kv_heads,
                 feedforward_hidden_dim, number_of_experts, number_of_active_experts,
                 dropout_rate, max_seq_len):
        super(TransformerBlock, self).__init__()

        assert number_of_query_heads % number_of_kv_heads == 0

        self.number_of_query_heads = number_of_query_heads
        self.number_of_kv_heads    = number_of_kv_heads
        self.queries_per_kv_head   = number_of_query_heads // number_of_kv_heads
        self.head_dim              = embedding_dim // number_of_query_heads
        self.embedding_dim         = embedding_dim

        self.rms_norm_before_attention   = RMSNorm(embedding_dim)
        self.rms_norm_before_feedforward = RMSNorm(embedding_dim)

        self.query_projection  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        kv_projection_dim      = number_of_kv_heads * self.head_dim
        self.key_projection    = nn.Linear(embedding_dim, kv_projection_dim, bias=False)
        self.value_projection  = nn.Linear(embedding_dim, kv_projection_dim, bias=False)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout_rate)

        # MoE replaces the dense SwiGLU FFN
        self.mixture_of_experts = MixtureOfExperts(
            embedding_dim=embedding_dim,
            feedforward_hidden_dim=feedforward_hidden_dim,
            number_of_experts=number_of_experts,
            number_of_active_experts=number_of_active_experts
        )

        cos_table, sin_table = compute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos_table)
        self.register_buffer('rope_sin', sin_table)

    def forward(self, token_representations, causal_mask):
        batch_size = token_representations.shape[0]
        seq_len    = token_representations.shape[1]

        # ---- GQA with RoPE (unchanged from Project 10) ----
        normed = self.rms_norm_before_attention(token_representations)

        Q = self.query_projection(normed)
        K = self.key_projection(normed)
        V = self.value_projection(normed)

        Q = Q.view(batch_size, seq_len, self.number_of_query_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.number_of_kv_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.number_of_kv_heads, self.head_dim)

        Q = apply_rope(Q, self.rope_cos, self.rope_sin)
        K = apply_rope(K, self.rope_cos, self.rope_sin)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        K_expanded = K.repeat_interleave(self.queries_per_kv_head, dim=1)
        V_expanded = V.repeat_interleave(self.queries_per_kv_head, dim=1)

        attention_scores  = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores  = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attention_output  = torch.matmul(attention_weights, V_expanded)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
        attention_output = self.output_projection(attention_output)

        token_representations = token_representations + attention_output

        # ---- MoE feed forward ----
        normed = self.rms_norm_before_feedforward(token_representations)
        moe_output, load_balancing_loss = self.mixture_of_experts(normed)
        token_representations = token_representations + moe_output

        return token_representations, load_balancing_loss


# ============================================================
# MINI LANGUAGE MODEL WITH MoE
# ============================================================

class MiniLanguageModel(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, number_of_query_heads,
                 number_of_kv_heads, feedforward_hidden_dim, number_of_blocks,
                 number_of_experts, number_of_active_experts,
                 dropout_rate, max_sequence_length):
        super(MiniLanguageModel, self).__init__()

        self.word_embedding    = nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                number_of_query_heads=number_of_query_heads,
                number_of_kv_heads=number_of_kv_heads,
                feedforward_hidden_dim=feedforward_hidden_dim,
                number_of_experts=number_of_experts,
                number_of_active_experts=number_of_active_experts,
                dropout_rate=dropout_rate,
                max_seq_len=max_sequence_length
            )
            for _ in range(number_of_blocks)
        ])

        self.final_rms_norm    = RMSNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size, bias=False)

    def _build_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, token_indices):
        seq_len               = token_indices.shape[1]
        token_representations = self.word_embedding(token_indices)
        token_representations = self.embedding_dropout(token_representations)
        causal_mask           = self._build_causal_mask(seq_len, token_indices.device)

        total_load_balancing_loss = torch.tensor(0.0, device=token_indices.device)

        for transformer_block in self.transformer_blocks:
            token_representations, block_load_balancing_loss = transformer_block(
                token_representations, causal_mask
            )
            total_load_balancing_loss = total_load_balancing_loss + block_load_balancing_loss

        # Average load balancing loss across blocks
        total_load_balancing_loss = total_load_balancing_loss / len(self.transformer_blocks)

        token_representations     = self.final_rms_norm(token_representations)
        last_token_representation = token_representations[:, -1, :]
        output_scores             = self.output_projection(last_token_representation)

        return output_scores, total_load_balancing_loss


# ---- Initialise model ----

embedding_dim             = 64
number_of_query_heads     = 4
number_of_kv_heads        = 2
feedforward_hidden_dim    = 128
number_of_blocks          = 4
number_of_experts         = 8
number_of_active_experts  = 2
auxiliary_loss_weight     = 0.01   # weight for load balancing loss
dropout_rate              = 0.1
learning_rate             = 0.001
number_of_epochs          = 2000

model = MiniLanguageModel(
    vocabulary_size=vocabulary_size,
    embedding_dim=embedding_dim,
    number_of_query_heads=number_of_query_heads,
    number_of_kv_heads=number_of_kv_heads,
    feedforward_hidden_dim=feedforward_hidden_dim,
    number_of_blocks=number_of_blocks,
    number_of_experts=number_of_experts,
    number_of_active_experts=number_of_active_experts,
    dropout_rate=dropout_rate,
    max_sequence_length=sequence_length
).to(device)

loss_function = nn.CrossEntropyLoss()
optimiser     = optim.Adam(model.parameters(), lr=learning_rate)
scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=number_of_epochs)

total_parameters        = sum(p.numel() for p in model.parameters())
active_params_per_token = total_parameters - (
    number_of_blocks * number_of_experts *
    3 * embedding_dim * feedforward_hidden_dim *
    (1 - number_of_active_experts / number_of_experts)
)

print(f"Total parameters:          {total_parameters:,}")
print(f"Project 10 had:            ~150,000")
print(f"Active experts per token:  {number_of_active_experts} of {number_of_experts} ({number_of_active_experts/number_of_experts:.0%})")


# ---- Training loop with combined loss ----

training_loss_history              = []
load_balancing_loss_history        = []

for epoch in range(number_of_epochs):
    model.train()
    total_loss              = 0
    total_load_balance_loss = 0
    num_batches             = 0

    for batch_sequences, batch_targets in training_loader:
        batch_sequences = batch_sequences.to(device)
        batch_targets   = batch_targets.to(device)

        optimiser.zero_grad()

        output_scores, load_balancing_loss = model(batch_sequences)

        # Combined loss: cross-entropy + weighted load balancing
        cross_entropy_loss = loss_function(output_scores, batch_targets)
        combined_loss      = cross_entropy_loss + auxiliary_loss_weight * load_balancing_loss

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss              += cross_entropy_loss.item()
        total_load_balance_loss += load_balancing_loss.item()
        num_batches             += 1

    scheduler.step()
    average_ce_loss  = total_loss / num_batches
    average_lb_loss  = total_load_balance_loss / num_batches
    training_loss_history.append(average_ce_loss)
    load_balancing_loss_history.append(average_lb_loss)

    if epoch % 400 == 0:
        print(f"Epoch {epoch:5d}  "
              f"ce_loss: {average_ce_loss:.4f}  "
              f"lb_loss: {average_lb_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")


# ---- Text generation ----

def generate_text(seed_text, number_of_tokens_to_generate=16, temperature=0.8):
    model.eval()
    generated_ids = bpe_tokenizer.encode(seed_text.lower()).ids.copy()

    with torch.no_grad():
        for _ in range(number_of_tokens_to_generate):
            context_ids     = generated_ids[-sequence_length:]
            sequence_tensor = torch.tensor(context_ids).unsqueeze(0).to(device)
            output_scores, _ = model(sequence_tensor)

            if temperature == 0.0:
                predicted_id = torch.argmax(output_scores, dim=-1).item()
            else:
                probabilities = torch.softmax(output_scores / temperature, dim=-1)
                predicted_id  = torch.multinomial(probabilities, num_samples=1).item()

            generated_ids.append(predicted_id)

    decoded = bpe_tokenizer.decode(generated_ids)
    return ' '.join(decoded.split())


print()
print("Generated text (temperature=0.8):")
print(" ", generate_text("the sky is cloudy"))
print(" ", generate_text("bring your umbrella"))
print(" ", generate_text("dark clouds mean"))
print(" ", generate_text("the rain will"))
print(" ", generate_text("a clear sky"))

print(f"\nFinal cross-entropy loss:  {training_loss_history[-1]:.4f}")
print(f"Final load balancing loss: {load_balancing_loss_history[-1]:.4f}")
print(f"  (1.0 = perfectly uniform, higher = imbalanced)")


# ---- Plot both loss curves ----

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(training_loss_history, color='steelblue', linewidth=1.5,
             label=f'MoE ({number_of_experts} experts, top-{number_of_active_experts})')
axes[0].axhline(
    y=math.log(vocabulary_size), color='tomato',
    linestyle='--', linewidth=1,
    label=f'Random baseline: {math.log(vocabulary_size):.2f}'
)
axes[0].set_title('Cross-Entropy Loss', fontsize=13)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].legend(fontsize=10)

axes[1].plot(load_balancing_loss_history, color='darkorange', linewidth=1.5,
             label='Load balancing loss')
axes[1].axhline(y=1.0, color='steelblue', linestyle='--', linewidth=1,
                label='1.0 = perfectly uniform')
axes[1].set_title('Load Balancing Loss (Expert Utilisation)', fontsize=13)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(fontsize=10)

plt.suptitle(f'Mini LLM: Mixture of Experts ({number_of_experts} experts, top-{number_of_active_experts})',
             fontsize=14)
plt.tight_layout()
plt.savefig('loss_curve_moe.png', dpi=150)
plt.show()

print("Loss curves saved to loss_curve_moe.png")