import argparse
import os
import json
import torch
from training.utils import Vocab
from training.train import RNNClassifier, LSTMClassifier, SelfAttentionClassifier

def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu')
    return ck

def build_vocab_from_stoi(stoi):
    v = Vocab()
    v.stoi = dict(stoi)
    # build itos list
    size = max(v.stoi.values()) + 1
    itos = [None] * size
    for tok, idx in v.stoi.items():
        if idx < size:
            itos[idx] = tok
    v.itos = [t if t is not None else '<unk>' for t in itos]
    return v

def make_model(model_type, vocab_size, args, num_classes):
    if model_type == 'rnn':
        return RNNClassifier(vocab_size, args.embed_dim, args.hidden_dim, num_classes)
    elif model_type == 'lstm':
        return LSTMClassifier(vocab_size, args.embed_dim, args.hidden_dim, num_classes)
    else:
        return SelfAttentionClassifier(vocab_size, args.embed_dim, args.num_heads, args.hidden_dim, num_classes)

def predict_texts(model_type, checkpoint, texts, args):
    ck = load_checkpoint(checkpoint)
    stoi = ck.get('vocab')
    if stoi is None:
        raise RuntimeError('vocab not found in checkpoint')
    vocab = build_vocab_from_stoi(stoi)
    num_classes = ck.get('model_state', {}).get(next(iter(ck.get('model_state'))), None)
    # num_classes not stored reliably; instead infer from classifier fc
    # construct model
    model = make_model(model_type, len(vocab), args, num_classes=2)
    model.load_state_dict(ck['model_state'])
    model.eval()

    results = []
    for t in texts:
        ids = vocab.encode(t)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        src = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(src, lengths)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
        results.append({'text': t, 'pred': pred, 'probs': probs.squeeze().tolist()})
    return results

def export_checkpoint(checkpoint_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ck = load_checkpoint(checkpoint_path)
    base = os.path.basename(checkpoint_path)
    out_path = os.path.join(out_dir, base)
    torch.save(ck, out_path)
    # write metadata
    meta = {'exported_from': checkpoint_path}
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['rnn','lstm','attn'], required=True)
    p.add_argument('--checkpoint', default=None, help='Path to checkpoint (defaults to training/models/best_<model>.pt)')
    p.add_argument('--text', help='Single text to predict')
    p.add_argument('--file', help='File with one text per line')
    p.add_argument('--export-dir', help='If set, copy checkpoint and metadata into this folder')
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--num-heads', type=int, default=4)
    args = p.parse_args()

    ck = args.checkpoint or os.path.join('training','models', f'best_{args.model}.pt')
    if not os.path.exists(ck):
        raise FileNotFoundError(f'Checkpoint not found: {ck}')

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
    if not texts:
        texts = ["Example query about security controls."]

    res = predict_texts(args.model, ck, texts, args)
    print(json.dumps(res, indent=2))

    if args.export_dir:
        out = export_checkpoint(ck, args.export_dir)
        print('Exported checkpoint to', out)

if __name__ == '__main__':
    main()
