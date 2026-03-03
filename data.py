

import os
import urllib.request
import zipfile
from collections import Counter

import numpy as np


class DataProcessor:

    def __init__(self, min_freq=10, subsample_t=1e-4, max_words=None):
        self.min_freq = min_freq
        self.subsample_t = subsample_t
        self.max_words = max_words

        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freqs = None       # normalized frequency per word
        self.sampling_probs = None   # negative sampling distribution
        self.discard_probs = None    # subsampling discard probabilities
        self.sentences = []

    def download_text8(self):

        if not os.path.exists("text8"):
            print("Downloading text8 dataset...", flush=True)
            urllib.request.urlretrieve(
                "http://mattmahoney.net/dc/text8.zip", "text8.zip"
            )
            with zipfile.ZipFile("text8.zip") as z:
                z.extractall()
            os.remove("text8.zip")

        with open("text8", "r") as f:
            text = f.read()

        words = text.strip().split()

        #  limit how many words we use
        if self.max_words is not None:
            words = words[:self.max_words]

        print(f"Total words loaded: {len(words)}", flush=True)
        return words

    def build_vocab(self, words):
        """Count word frequencies and build word <-> index mappings."""
        counts = Counter(words)

        # only keep words that appear often enough
        vocab = sorted([w for w, c in counts.items() if c >= self.min_freq])

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

      
        total = sum(counts[w] for w in vocab)
        self.word_freqs = np.zeros(self.vocab_size)
        for w in vocab:
            self.word_freqs[self.word2idx[w]] = counts[w] / total

        print(f"Vocabulary size: {self.vocab_size}", flush=True)

    def compute_subsampling_probs(self):

        t = self.subsample_t
        self.discard_probs = 1.0 - np.sqrt(t / (self.word_freqs + 1e-10))
        self.discard_probs = np.clip(self.discard_probs, 0, 1)

    def compute_negative_sampling_probs(self):

        powered = self.word_freqs ** 0.75
        self.sampling_probs = powered / powered.sum()

    def prepare_sentences(self, words, chunk_size=1000):

        encoded = [self.word2idx[w] for w in words if w in self.word2idx]

        self.sentences = []
        for start in range(0, len(encoded), chunk_size):
            chunk = encoded[start:start + chunk_size]

            # randomly discard frequent words
            filtered = [w for w in chunk
                        if np.random.random() > self.discard_probs[w]]

            if len(filtered) >= 2:
                self.sentences.append(filtered)

        print(f"Sentences (chunks): {len(self.sentences)}", flush=True)

    def generate_training_pairs(self, window_size=5):

        pairs = []
        for sent in self.sentences:
            for i in range(len(sent)):
                center = sent[i]
                left = max(0, i - window_size)
                right = min(len(sent), i + window_size + 1)

                for j in range(left, right):
                    if j != i:
                        pairs.append((center, sent[j]))

        pairs = np.array(pairs)
        print(f"Training pairs: {len(pairs)}", flush=True)
        return pairs

    def process(self, window_size=5):

        words = self.download_text8()
        self.build_vocab(words)
        self.compute_subsampling_probs()
        self.compute_negative_sampling_probs()
        self.prepare_sentences(words)
        pairs = self.generate_training_pairs(window_size)
        return pairs