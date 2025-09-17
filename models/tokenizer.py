# models/tokenizer.py
class CharTokenizer:
    """
    Simple char-level tokenizer for code/conversation.
    """
    def __init__(self, chars=None):
        if chars is None:
            chars = list("abcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}+-*/=_<> \n")
        self.chars = chars
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}  # 0 for padding
        self.itos = {i+1: ch for i, ch in enumerate(chars)}
        self.pad_token = 0

    def encode(self, text):
        return [self.stoi.get(ch, self.pad_token) for ch in text.lower()]

    def decode(self, tokens):
        return ''.join(self.itos.get(t, '') for t in tokens if t != self.pad_token)

    @property
    def vocab_size(self):
        return len(self.chars) + 1
