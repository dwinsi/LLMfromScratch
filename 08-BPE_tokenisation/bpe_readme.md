# 08: BPE Tokenisation

Every project so far has split text into words. "the sky is cloudy" becomes four tokens: `the`, `sky`, `is`, `cloudy`. This is called **word-level tokenisation** and it is the simplest possible approach.

Real language models do not work this way. GPT-4, LLaMA, Gemma, and every other modern model use a different strategy called **Byte Pair Encoding (BPE)**. This project replaces the word-level tokeniser with a BPE tokeniser trained on the same corpus. Everything else, the Transformer blocks, training loop, cosine annealing, batching, stays identical to Project 7. The only change is how text is converted into numbers.

---

## What is wrong with word-level tokenisation?

Word-level tokenisation has three practical problems that become serious at scale.

### Problem 1: out-of-vocabulary words

A word-level tokeniser can only handle words it has seen during training. Every other word becomes `[UNK]` (unknown). If the model was trained on English and encounters the word "supercalifragilistic", it cannot represent it at all. The information is simply lost.

For our small corpus with 198 unique words this is manageable. For a real language model trained on billions of web pages containing technical jargon, proper nouns, URLs, code, and dozens of languages, a pure word-level vocabulary would need millions of entries and still fail constantly on rare or new words.

### Problem 2: vocabulary size explodes with scale

English has hundreds of thousands of words, and that is before counting names, place names, technical terms, different languages, and intentional misspellings. A word-level vocabulary large enough to handle real-world text would be enormous. The embedding table (one row per vocabulary item) and the output projection layer (one column per vocabulary item) both scale directly with vocabulary size. A vocabulary of one million words would create embedding and output matrices too large to train efficiently.

### Problem 3: no sharing of information between related words

"run", "runs", "running", "runner" are all different entries in a word-level vocabulary. The model treats them as four completely unrelated tokens. Any information learned about "run" does not transfer to "running". BPE handles this naturally because all four words share the same learned sub-pieces.

---

## What BPE does instead

BPE represents text as sequences of **subword tokens**: pieces of words rather than whole words.

The key insight is that most words are built from recognisable parts. The word "running" contains "run" and "ning". "unhappy" contains "un" and "happy". "tokenisation" contains "token", "isa", "tion". A tokeniser that can identify and reuse these parts gets the best of both worlds: a manageable vocabulary size, and the ability to handle any word, even ones never seen before, by breaking them into known pieces.

A BPE vocabulary typically contains:

- Individual characters and bytes (so any text is representable, no unknown tokens)
- Common character combinations that appear frequently
- Frequent whole words or word fragments
- Special tokens like `[BOS]` (beginning of sequence), `[EOS]` (end), `[UNK]` (unknown), `[PAD]` (padding)

GPT-2 uses a vocabulary of 50,257 BPE tokens. LLaMA 3 uses 128,256. Our small model uses 256, which is tiny but enough to demonstrate the concept clearly.

---

## How BPE is trained: the merge algorithm

BPE builds its vocabulary by starting small and merging frequently co-occurring pairs.

### Step 1: start with individual bytes

The alphabet begins as every individual byte (0 to 255). Every possible piece of text can be represented as a sequence of bytes, so the tokeniser can always handle any input, even unusual characters, with zero unknown tokens.

### Step 2: count all adjacent pairs

Scan the entire training corpus and count how many times each adjacent pair of tokens appears next to each other.

For example, in a corpus containing many English words, the pair `(t, h)` would appear very frequently because "the", "this", "that", "there" all start with "th".

### Step 3: merge the most frequent pair

Take the most frequently occurring pair and merge it into a single new token. `t` + `h` becomes `th`. Add this new token to the vocabulary.

### Step 4: repeat

Re-scan the corpus (now with the new merged token in place), find the next most frequent pair, merge it, add it to the vocabulary. Repeat until the vocabulary reaches the target size.

```text
Starting vocabulary:  {t, h, e, s, k, y, ...}    (individual bytes)

After iteration 1:    {t, h, e, s, k, y, ..., th}   ("th" is very frequent)
After iteration 2:    {t, h, e, s, k, y, ..., th, the}
After iteration 3:    {t, h, e, s, k, y, ..., th, the, is}
...
After 252 iterations: {256 tokens total}
```

The final vocabulary captures the most common pieces of the language in the training corpus. Common short words become single tokens. Common prefixes and suffixes become tokens. Rare or long words get split into their most common sub-pieces.

---

## BPE in practice with the HuggingFace tokenizers library

Training and using a BPE tokeniser takes about ten lines of code with the `tokenizers` library:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Create the tokeniser structure
bpe_tokenizer               = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = ByteLevel()
bpe_tokenizer.decoder       = ByteLevelDecoder()

# Train on the corpus
bpe_trainer = BpeTrainer(
    vocab_size=256,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    min_frequency=1
)
bpe_tokenizer.train(files=["weather_corpus_v2.txt"], trainer=bpe_trainer)
bpe_tokenizer.save("weather_bpe_tokenizer.json")
```

**`ByteLevel` pre-tokeniser and decoder:** before running the BPE merge algorithm, the text is first split at byte boundaries and spaces are encoded as a special `Ġ` character (the letter G with a cedilla above). This is the same approach GPT-2 uses. It ensures that the boundary between words is preserved in the tokens and that decoding always reconstructs the original text correctly.

**`min_frequency=1`:** a token pair must appear at least this many times in the corpus to be eligible for merging. On a small corpus like ours, setting this to 1 means any pair that appears even once can be merged, which maximises vocabulary coverage.

**`vocab_size=256`:** the trainer will run the merge algorithm until the vocabulary reaches 256 tokens. This is the target size, not the starting size.

---

## What the tokeniser produces

Let us see what the trained tokeniser does with a few example sentences:

```text
Input:  "the sky is cloudy today"
Tokens: ['the', 'Ġsky', 'Ġis', 'Ġcloud', 'y', 'Ġtoday']
IDs:    [34, 107, 58, 113, 87, 95]

Input:  "bring your umbrella when it rains"
Tokens: ['bring', 'Ġyour', 'Ġumbrella', 'Ġwhen', 'Ġit', 'Ġra', 'ins']
IDs:    [52, 131, 178, 103, 57, 99, 76]
```

Notice a few things:

`Ġ` (shown as a prefix on most words) represents a space before that word. This is how the ByteLevel tokeniser encodes word boundaries inside the token string itself. When you decode the tokens, the `Ġ` characters are automatically converted back to spaces, so you always get clean readable text out.

"cloudy" is split into `cloud` and `y` because `cloudy` as a whole does not appear often enough in the corpus to have earned its own merge, but `cloud` does appear frequently and gets its own token.

"rains" is split into `ra` and `ins`. The merge algorithm found `ra` and `ins` to be useful sub-pieces based on frequency in the corpus.

"umbrella" is kept as a single token because it appears frequently enough in this weather corpus to have earned its own entry.

---

## Building training sequences from token IDs

Once the corpus is encoded into token IDs, the process of building training sequences is identical to Projects 4 through 7: a sliding window moves through the token ID sequence, and each window of 8 tokens predicts the 9th.

```python
# Encode the entire corpus as a flat list of token IDs
encoded_corpus = bpe_tokenizer.encode(corpus_text)
all_token_ids  = encoded_corpus.ids

# 80/20 training/validation split at the token level
split_idx = int(0.8 * len(all_token_ids))
train_ids = all_token_ids[:split_idx]
val_ids   = all_token_ids[split_idx:]

# Sliding window: 8 tokens predict the 9th
for i in range(len(train_ids) - sequence_length):
    training_sequences.append(train_ids[i : i + sequence_length])
    training_targets.append(train_ids[i + sequence_length])
```

The sequence length increases from 4 (Project 7) to 8 here. BPE tokens are typically shorter than words, so an 8-token context window covers roughly the same amount of text as a 4-word window. It also gives the model slightly more context per prediction step.

---

## DataLoader: proper batching

Project 7 built batching manually using `torch.randperm`. This project uses PyTorch's built-in `DataLoader`:

```python
from torch.utils.data import TensorDataset, DataLoader

training_dataset = TensorDataset(sequences_tensor, targets_tensor)
training_loader  = DataLoader(training_dataset, batch_size=32, shuffle=True)
```

`TensorDataset` bundles the input sequences and target labels into a single dataset object. `DataLoader` wraps it with automatic batching, shuffling, and iteration. Setting `shuffle=True` randomises the order of examples each epoch, which is equivalent to what `torch.randperm` did manually before.

The training loop becomes cleaner:

```python
for batch_sequences, batch_targets in training_loader:
    batch_sequences = batch_sequences.to(device)
    batch_targets   = batch_targets.to(device)

    optimiser.zero_grad()
    output_scores = model(batch_sequences)
    loss          = loss_function(output_scores, batch_targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
```

The model architecture, loss function, optimiser, gradient clipping, and cosine annealing are all unchanged from Project 7.

---

## What changes in the model

The model architecture is identical to Project 7 with one difference: the embedding table and output projection now use the BPE vocabulary size (256) instead of the word vocabulary size (198).

```text
Project 7 (word-level):
  word_embedding_matrix    198 x 64    = 12,672 parameters
  output_projection         64 x 198   = 12,672 parameters

Project 8 (BPE):
  word_embedding_matrix    256 x 64    = 16,384 parameters
  output_projection         64 x 256   = 16,384 parameters
```

Everything inside the Transformer blocks stays the same: 4 blocks, 64-dimensional embeddings, 4 attention heads, 128-dimensional feed-forward layers.

---

## Text generation with BPE

Generating text with BPE differs from word-level generation in two ways: the seed text must be encoded to token IDs first, and the generated token IDs must be decoded back to text at the end.

```python
def generate_text(seed_text, number_of_tokens_to_generate=16, temperature=0.8):
    model.eval()

    # Encode the seed using the BPE tokeniser
    encoded_seed  = bpe_tokenizer.encode(seed_text.lower())
    generated_ids = encoded_seed.ids.copy()

    with torch.no_grad():
        for _ in range(number_of_tokens_to_generate):
            context_ids     = generated_ids[-sequence_length:]
            sequence_tensor = torch.tensor(context_ids).unsqueeze(0).to(device)
            output_scores   = model(sequence_tensor)

            probabilities = torch.softmax(output_scores / temperature, dim=-1)
            predicted_id  = torch.multinomial(probabilities, num_samples=1).item()
            generated_ids.append(predicted_id)

    # Decode all token IDs back to clean text
    decoded = bpe_tokenizer.decode(generated_ids)
    return ' '.join(decoded.split())
```

`bpe_tokenizer.decode()` handles the conversion from token IDs back to readable text, including converting `Ġ` back to spaces and `Ċ` back to newlines. The `' '.join(decoded.split())` at the end normalises any multiple spaces or newlines into single spaces.

Sample output:

```text
the sky is cloudy -> today bring your umbrella when it rains dark clouds
bring your umbrella -> when it rains dark clouds mean heavy rain the weather
dark clouds mean -> heavy rain the weather looks wet outside a clear sky
the rain will stop -> by evening sunny weather makes people happy dark clouds
a clear sky means -> no rain wind and clouds bring storms heavy rain is coming
```

---

## Word-level vs BPE: a direct comparison

```text
Property                     Word-level (Project 7)    BPE (Project 8)
---------------------------------------------------------------------
Vocabulary size                   198                       256
Sequence length                     4 words                   8 tokens
Handles unseen words              No (becomes [UNK])       Yes (splits into subpieces)
Shares info across "run/running"  No                        Yes (shared subwords)
Tokeniser library                 Built by hand             HuggingFace tokenizers
Training sequences                ~399                      ~560
```

The BPE tokeniser produces more training sequences from the same corpus (more tokens per sentence = more sliding windows) and handles any input text without unknown tokens.

---

## Why this matters for real language models

At the scale of GPT-4 or LLaMA 3:

- Training data contains hundreds of billions of words across dozens of languages, plus code, math, and web markup
- Word-level tokenisation would require a vocabulary of millions of entries to cover even common text
- BPE with ~100,000 tokens covers essentially all English text cleanly, handles most other languages, and degrades gracefully on rare input by splitting it into known sub-pieces

The same BPE tokeniser trained once is reused for both training and inference. Any text passed to the model at inference time is encoded with the same tokeniser, so the model always receives input in the exact format it was trained on.

The vocabulary size (256 here, ~50,000 to 130,000 in production models) is one of the most consequential choices in building a language model. It directly affects:

- Embedding table size and output projection size (both scale with vocabulary)
- How many tokens are needed to represent a piece of text (larger vocab = shorter sequences = faster training)
- How well the model generalises to rare words and new languages

---

## Running

```bash
pip install torch tokenizers matplotlib
python 08_mini_llm_bpe.py
```

The script trains the BPE tokeniser on the corpus, prints sample encodings, trains the model for 2000 epochs, generates text samples, and saves the loss curve to `loss_curve_bpe.png`.

## Files

```text
08_mini_llm_bpe.py          full training script with BPE tokenisation
weather_corpus_v2.txt       91-sentence training corpus (same as Project 7)
config.json                 hyperparameters (vocab size, sequence length, epochs)
weather_bpe_tokenizer.json  saved tokeniser (generated on first run)
loss_curve_bpe.png          training vs validation loss
```
