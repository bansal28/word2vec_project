# Word2Vec Skip-Gram with Negative Sampling — Pure NumPy

Implementation of the Skip-Gram word embedding model with Negative Sampling (SGNS), built from scratch using only NumPy. No ML frameworks (PyTorch, TensorFlow, etc.) are used.

## References

- **Architecture (Skip-Gram):** Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013), arXiv:1301.3781
- **Training method (Negative Sampling, Subsampling):** Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NIPS 2013), arXiv:1310.4546

## How to Run

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Mac/Linux
# .venv\Scripts\activate    # on Windows

pip install numpy

python main.py
```

The script automatically downloads the text8 dataset (~31MB) on first run. Training takes about 19 minutes (3 epochs, ~23M pairs) on an M1 MacBook Air.

## Project Structure

```
word2vec/
├── data.py    — DataProcessor class: loading, vocab, subsampling, pair generation
├── model.py   — Word2Vec class: forward pass, backward pass, SGD update, evaluation
├── main.py    — Entry point: hyperparameters, training loop, evaluation calls
└── README.md
```

## Dataset

**text8** — a cleaned subset of English Wikipedia (~100MB, ~17M words). We use the first 5 million words. It's pre-lowercased with no punctuation, which means no preprocessing is needed beyond tokenization by whitespace.

This is the standard benchmark dataset used for word2vec experiments.

## Algorithm

### Skip-Gram Model

The core idea: given a center word, predict the words that appear near it (context words). Words that frequently appear in similar contexts end up with similar vector representations.

For each word in the corpus, we look at a window of surrounding words and create (center, context) training pairs. For example, in the sentence `the cat sat on mat` with window=2, the word `sat` creates pairs: (sat, the), (sat, cat), (sat, on), (sat, mat).

### Negative Sampling

Instead of predicting over the entire vocabulary (expensive softmax), we simplify the problem: for each real (center, context) pair, sample k random "negative" words and train the model to distinguish real pairs from fake ones.

The loss function for one training example:

```
L = -log(σ(v_c · v_o)) - Σ_k log(σ(-v_c · v_k))
```

Where:
- `σ` is the sigmoid function: σ(x) = 1 / (1 + e^(-x))
- `v_c` = center word vector (from W_in)
- `v_o` = positive context word vector (from W_out)
- `v_k` = negative sample vectors (from W_out)

The first term pushes the dot product of real pairs high. The second term pushes the dot product of fake pairs low.

### Gradient Derivation

Starting from the loss:

```
L = -log(σ(v_c · v_o)) - Σ_k log(σ(-v_c · v_k))
```

Using the chain rule and the property that d/dx log(σ(x)) = 1 - σ(x):

Let p = σ(v_c · v_o) and q_k = σ(-v_c · v_k), then:

**Gradient w.r.t. context vector v_o** (only the first term involves v_o):

```
∂L/∂v_o = (p - 1) · v_c
```

When p ≈ 1 (correct prediction), gradient is near zero. When p ≈ 0 (wrong), gradient is large. This makes sense — we only update when the model is wrong.

**Gradient w.r.t. each negative vector v_k** (only the second term involves v_k):

```
∂L/∂v_k = (1 - q_k) · v_c
```

**Gradient w.r.t. center vector v_c** (both terms involve v_c):

```
∂L/∂v_c = (p - 1) · v_o + Σ_k (1 - q_k) · v_k
```

Parameters are updated with SGD: `w ← w - lr · ∂L/∂w`

### Key Implementation Details

**Two embedding matrices:** W_in (center word embeddings) and W_out (context word embeddings). Using separate matrices avoids a word being its own best match. After training, W_in is used as the final embeddings.

**Subsampling of frequent words** (from the NIPS 2013 paper, Section 2.3): Very common words like "the", "a", "of" are randomly discarded during training with probability `P(discard) = 1 - sqrt(t / freq)` where t = 1e-4. This removes noisy, uninformative pairs and speeds up training.

**Negative sampling distribution** (from the NIPS 2013 paper): Negatives are sampled proportional to `freq^(3/4)` rather than raw frequency. The 3/4 power boosts rare words so they appear as negatives more often, preventing the model from only learning about common words.

**Linear learning rate decay** (from the 2013 paper, Section 4.2): Learning rate starts at 0.025 and decays linearly to near zero over training. This gives large updates early (when everything is wrong) and fine-grained updates later (when embeddings are mostly correct).

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Embedding dim | 100 | Standard choice, balances quality vs speed |
| Window size | 5 | Captures sufficient context |
| Epochs | 3 | Paper uses 3; diminishing returns after that |
| Negatives | 5 | Paper recommends 5-20 for small datasets |
| Learning rate | 0.025 → 0.0001 | Linear decay, following the paper |
| Min word freq | 10 | Filters rare words that add noise |
| Subsample threshold | 1e-4 | Aggressively drops stop words |

## Results

Training on 5M words from text8 (vocab: 23,599 words, 23M training pairs, 3 epochs, ~19 min total):

**Loss curve:** 4.16 → 2.58 → 2.36 → 2.31

### Nearest Neighbors

```
queen   → elizabeth, victoria, buckingham, princess, monarch
france  → spain, netherlands, italy, switzerland, nantes
dog     → hound, cat, eyed, ass, dogs
computer→ computers, computing, hardware, microcomputer, laptop
good    → bad, homework, stuff, worry, charm
```

### Word Analogies

```
france : paris  ::  germany : berlin     ✓
good   : better ::  bad     : worse      ✓
man    : king   ::  woman   : daughter   (partially — royalty-related but not queen)
```

## What I Would Improve

- Use the full text8 corpus (17M words) or a larger dataset for better embeddings
- Implement the random window trick from the paper (sample R from 1 to C instead of fixed window) to give closer words more weight
- Add multi-threading for faster training
- Save/load trained embeddings to avoid retraining
