# Word2Vec Skip-Gram with Negative Sampling — Using NumPy

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

The script automatically downloads the text8 dataset (~31MB) on first run. Training takes about 19 minutes (3 epochs, ~23M pairs).

## Project Structure

```
word2vec_project/
├── data.py    — DataProcessor class: loading, vocab, subsampling, pair generation
├── model.py   — Word2Vec class: forward pass, backward pass, SGD update, evaluation
├── main.py    — Entry point: hyperparameters, training loop, evaluation calls
└── README.md
```

## Dataset

**text8** — a cleaned subset of English Wikipedia (~100MB, ~17M words). We use the first 5 million words. It's pre-lowercased with no punctuation.

## Algorithm

### Skip-Gram Model

The core idea: given a center word, predict the words that appear near it (context words). Words that frequently appear in similar contexts end up with similar vector representations.

### Negative Sampling

Instead of predicting over the entire vocabulary (expensive softmax), we simplify the problem: for each real (center, context) pair, sample k random "negative" words and train the model to distinguish real pairs from fake ones.


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

**Loss curve:** 3.5 → 2.58 → 2.36 → 2.3

<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/adb2f953-32ca-4791-9266-c8fba0314ae3" />
<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/0be155cb-e540-4559-8bb3-cbab0bc956c8" />
<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/f2c139bd-baa4-4625-8cae-18692b6d37fc" />




