# data/convo_dataset.py
import random
import torch
from torch.utils.data import Dataset

class ConvoDataset(Dataset):
    """
    Small synthetic conversation dataset for next-token prediction.
    """
    def __init__(self, tokenizer, n_samples=10000, max_len=64):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_len = max_len
        self.pairs = [
            ("hello","hi"),
            ("what is 2+2?","4"),
            ("who are you?", "I am Akia, an ai model developed in Nairobi"),
            ("name of our model?","akia"),
            ("who made you?","I was made by Lacesse Ventures, a Kenyan Tech company"),
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        q,a = random.choice(self.pairs)
        text = q + " <sep> " + a
        tokens = self.tokenizer.encode(text)
        if len(tokens) < self.max_len:
            tokens += [0]*(self.max_len-len(tokens))
        else:
            tokens = tokens[:self.max_len]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        return tokens_tensor[:-1], tokens_tensor[1:]  # input, target
