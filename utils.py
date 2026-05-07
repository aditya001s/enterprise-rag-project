import re
from collections import Counter
import torch

def simple_tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return tokens

class Vocab:
    def __init__(self, max_size=20000, min_freq=1, unk_token='<unk>', pad_token='<pad>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.stoi = {pad_token:0, unk_token:1}
        self.itos = [pad_token, unk_token]

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(simple_tokenize(t))
        items = [w for w, c in counter.most_common(self.max_size) if c >= self.min_freq]
        for w in items:
            if w in self.stoi:
                continue
            self.stoi[w] = len(self.itos)
            self.itos.append(w)

    def encode(self, text):
        return [self.stoi.get(t, self.stoi[self.unk_token]) for t in simple_tokenize(text)]

    def __len__(self):
        return len(self.itos)

def collate_batch(batch):
    # batch: list of (tensor_ids, length, label)
    seqs, lengths, labels = zip(*batch)
    seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, labels
