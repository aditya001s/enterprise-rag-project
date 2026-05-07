import argparse
import os
from pathlib import Path
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import Vocab, collate_batch
from gpu_utils import get_device, clear_gpu_memory

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.vocab.encode(self.texts[idx])
        return seq, len(seq), int(self.labels[idx])

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):
            h = h[0]
        h = h[-1]
        return self.fc(h)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = h[-1]
        return self.fc(h)

class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq)
        emb = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        emb = emb.transpose(0, 1)  # seq, batch, embed
        mask = (x == 0)  # batch, seq -> True for padding
        out = self.transformer(emb, src_key_padding_mask=mask)
        out = out.transpose(0, 1)  # batch, seq, embed
        # mean pooling over valid tokens
        lens = lengths.unsqueeze(1).to(out.device).float()
        s = out.sum(dim=1)
        mean = s / lens.clamp(min=1.0)
        return self.fc(mean)

def train_epoch(model, dataloader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, lengths, y in tqdm(dataloader, desc='train'):
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, lengths, y in dataloader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss/total, correct/total

def main(args):
    df = pd.read_csv(args.data_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise RuntimeError("CSV must contain 'text' and 'label' columns")
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()
    labels = [int(l) for l in labels]
    num_classes = len(set(labels))
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

    vocab = Vocab(max_size=args.max_vocab)
    vocab.build(X_train)

    train_ds = TextDataset(X_train, y_train, vocab)
    val_ds = TextDataset(X_val, y_val, vocab)
    collate = collate_batch
    # use pin_memory and workers when using CUDA for faster transfers
    dl_kwargs = {}
    if torch.cuda.is_available() and args.use_cuda:
        dl_kwargs = {'pin_memory': True, 'num_workers': 4}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **dl_kwargs)

    device = get_device(use_gpu=args.use_cuda)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.model == 'rnn':
        model = RNNClassifier(len(vocab), args.embed_dim, args.hidden_dim, num_classes)
    elif args.model == 'lstm':
        model = LSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim, num_classes)
    else:
        model = SelfAttentionClassifier(len(vocab), args.embed_dim, args.num_heads, args.hidden_dim, num_classes)

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'vocab': vocab.stoi}, os.path.join(args.save_dir, f"best_{args.model}.pt"))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True, help='Path to CSV with text,label')
    p.add_argument('--model', choices=['rnn','lstm','attn'], default='lstm')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--max-vocab', type=int, default=20000)
    p.add_argument('--save-dir', default='./models')
    p.add_argument('--use-cuda', action='store_true')
    args = p.parse_args()
    main(args)
