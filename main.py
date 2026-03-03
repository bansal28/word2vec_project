
import numpy as np
from data import DataProcessor
from word2vec_model import Word2Vec


# hyperparameters 

EMBED_DIM = 100
WINDOW_SIZE = 5
N_EPOCHS = 3
N_NEGATIVES = 5
LEARNING_RATE = 0.025
MIN_LR = 0.0001
MIN_FREQ = 10
SUBSAMPLE_T = 1e-4
MAX_WORDS = 5000000   # first 5M words of text8 
SEED = 42

np.random.seed(SEED)


def train(model, pairs, data):

    total_pairs = len(pairs)
    total_steps = total_pairs * N_EPOCHS
    step = 0

    for epoch in range(N_EPOCHS):
        np.random.shuffle(pairs)
        epoch_loss = 0.0

        neg_samples = np.random.choice(
            data.vocab_size,
            size=(total_pairs, N_NEGATIVES),
            p=data.sampling_probs,
        )

        for i in range(total_pairs):
            center, context = pairs[i]
            negatives = neg_samples[i]

            # linear learning rate decay
            lr = LEARNING_RATE * (1 - step / total_steps)
            lr = max(lr, MIN_LR)

            # forward pass
            loss, c_vec, o_vec, k_vecs, ps, ns = model.forward(
                center, context, negatives
            )

            # backward pass
            g_c, g_o, g_k = model.backward(c_vec, o_vec, k_vecs, ps, ns)

            # SGD update
            model.update(center, context, negatives, g_c, g_o, g_k, lr)

            epoch_loss += loss
            step += 1

            # progress every 1m pairs
            if (i + 1) % 1000000 == 0:
                avg_loss = epoch_loss / (i + 1)

                print(
                    f"  Epoch {epoch+1} | {i+1}/{total_pairs} | "
                    f"loss: {avg_loss:.4f} | lr: {lr:.5f} | ",
                    flush=True,
                )


        print(
            f"Epoch {epoch+1} done | loss: {epoch_loss/total_pairs:.4f} | ",

            flush=True,
        )


def evaluate(model, data):
    print("\n Nearest Neighbor Evaluation", flush=True)
    test_words = [
        "king", "queen", "man", "woman",
        "france", "paris", "dog", "good", "bad", "computer",
    ]
    for word in test_words:
        model.most_similar(word, data.word2idx, data.idx2word, top_n=5)

    print("\n Analogy Tests ", flush=True)
    model.analogy("man", "king", "woman", data.word2idx, data.idx2word)
    model.analogy("france", "paris", "germany", data.word2idx, data.idx2word)
    model.analogy("good", "better", "bad", data.word2idx, data.idx2word)


def main():
    # step 1: prepare data
    print("=== Loading and processing data ===", flush=True)
    data = DataProcessor(
        min_freq=MIN_FREQ,
        subsample_t=SUBSAMPLE_T,
        max_words=MAX_WORDS,
    )
    pairs = data.process(window_size=WINDOW_SIZE)

    if len(pairs) == 0:
        print("No training pairs generated. Check your data settings.")
        return

    # step 2: create model
    print("\n=== Initializing model ===", flush=True)
    model = Word2Vec(data.vocab_size, embed_dim=EMBED_DIM)
    print(f"W_in shape:  {model.W_in.shape}")
    print(f"W_out shape: {model.W_out.shape}")

    # step 3: train
    print("\n=== Training ===", flush=True)
    train(model, pairs, data)

    # step 4: evaluate
    print("\n=== Evaluation ===", flush=True)
    evaluate(model, data)


if __name__ == "__main__":
    main()