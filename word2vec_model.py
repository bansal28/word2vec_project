
import numpy as np


class Word2Vec:

    def __init__(self, vocab_size, embed_dim=100):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.W_in = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim) )


        self.W_out = np.random.uniform( -0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))


    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    

    def forward(self, center, context, negatives):

        center_vec = self.W_in[center]        # (embed_dim,)
        context_vec = self.W_out[context]     # (embed_dim,)
        negative_vecs = self.W_out[negatives] # (k, embed_dim)

        # score the real pair — should be high
        pos_score = self.sigmoid(np.dot(center_vec, context_vec))

        # score the fake pairs — should be low (note the minus sign)
        neg_scores = self.sigmoid(-np.dot(negative_vecs, center_vec))

        # negative sampling loss (epsilon for numerical stability)
        loss = -np.log(pos_score + 1e-7) - np.sum(np.log(neg_scores + 1e-7))

        return loss, center_vec, context_vec, negative_vecs, pos_score, neg_scores

    def backward(self, center_vec, context_vec, negative_vecs, pos_score, neg_scores):

        # gradient for positive context word
        grad_context = (pos_score - 1) * center_vec

        # gradient for each negative sample
        grad_negatives = (1 - neg_scores)[:, np.newaxis] * center_vec[np.newaxis, :]

        # gradient for center word (combines both terms)
        grad_center = (pos_score - 1) * context_vec
        grad_center += np.sum(
            (1 - neg_scores)[:, np.newaxis] * negative_vecs, axis=0
        )

        return grad_center, grad_context, grad_negatives

    def update(self, center, context, negatives, grad_center, grad_context, grad_negatives, lr):
    
        self.W_in[center] -= lr * grad_center
        self.W_out[context] -= lr * grad_context
        self.W_out[negatives] -= lr * grad_negatives

    def most_similar(self, word, word2idx, idx2word, top_n=5):
        """Find closest words using cosine similarity on W_in embeddings."""
        if word not in word2idx:
            print(f"'{word}' not in vocabulary")
            return

        idx = word2idx[word]
        vec = self.W_in[idx]

        # cosine similarity = dot product of normalized vectors
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-8
        normalized = self.W_in / norms
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        scores = normalized @ vec_norm
        scores[idx] = -1  # don't match the word with itself

        best = np.argsort(scores)[::-1][:top_n]

        print(f"\nNearest to '{word}':")
        for i in best:
            print(f"  {idx2word[i]:<15} {scores[i]:.4f}")

    def analogy(self, a, b, c, word2idx, idx2word, top_n=5):
        """
        Solve: a is to b as c is to ?
        Example: man is to king as woman is to ? -> queen
        Method: vec(b) - vec(a) + vec(c), find nearest word
        """
        for w in [a, b, c]:
            if w not in word2idx:
                print(f"'{w}' not in vocabulary")
                return

        vec = (self.W_in[word2idx[b]]
               - self.W_in[word2idx[a]]
               + self.W_in[word2idx[c]])

        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-8
        normalized = self.W_in / norms
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        scores = normalized @ vec_norm

        # exclude the three input words
        for w in [a, b, c]:
            scores[word2idx[w]] = -1

        best = np.argsort(scores)[::-1][:top_n]

        print(f"\n'{a}' is to '{b}' as '{c}' is to:")
        for i in best:
            print(f"  {idx2word[i]:<15} {scores[i]:.4f}")