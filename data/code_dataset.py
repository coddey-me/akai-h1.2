# data/code_dataset.py
import random
import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    """
    Very small synthetic code dataset for next-token prediction.
    """
    def __init__(self, tokenizer, n_samples=10000, max_len=64):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_len = max_len
        self.snippets = [
            "def add(a,b): return a+b",
            "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
            "for i in range(10): print(i)",
            "if x>0: print('positive') else: print('nonpositive')",
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        text = random.choice(self.snippets)
        tokens = self.tokenizer.encode(text)
        if len(tokens) < self.max_len:
            tokens += [0]*(self.max_len-len(tokens))
        else:
            tokens = tokens[:self.max_len]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        return tokens_tensor[:-1], tokens_tensor[1:]  # input, target
