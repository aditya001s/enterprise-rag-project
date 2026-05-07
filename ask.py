#!/usr/bin/env python
#!/usr/bin/env python
import argparse
import sys
import subprocess
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from numpy.linalg import norm

def retrieve(query, emb, emb_norm, metas, chunks, model, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    qn = q_emb / (norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_norm @ qn[0]
    idxs = sims.argsort()[-k:][::-1]
    results = []
    for rank, idx in enumerate(idxs, 1):
        meta = metas.iloc[idx]
        text = chunks.iloc[idx]['text'] if 'text' in chunks.columns else ''
        snippet = text.replace('\n',' ')[:1000]
        results.append({'rank': rank, 'score': float(sims[idx]), 'source': meta.get('source'), 'chunk_id': int(meta.get('chunk_id')), 'snippet': snippet})
    return results

def call_model_predict(model_name, question):
    # call evaluate.py to get model prediction
    base = os.path.dirname(__file__)
    eval_script = os.path.join(base, 'evaluate.py')
    cmd = [sys.executable, eval_script, '--model', model_name, '--text', question]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = proc.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception:
            return out
    except subprocess.CalledProcessError as e:
        return {'error': e.stderr}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['rnn','lstm','attn'], default='attn', help='Model to query for a label (optional)')
    p.add_argument('--k', type=int, default=3, help='Number of passages to retrieve')
    p.add_argument('--no-model', action='store_true', help='Do not call the classifier model')
    args = p.parse_args()

    base = os.path.dirname(__file__)
    emb_path = os.path.join(base, 'embeddings.npy')
    metas_path = os.path.join(base, 'metas.json')
    chunks_path = os.path.join(base, 'extracted_chunks.csv')

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings file not found at: {emb_path}. Run embed.py to generate it.")
    if not os.path.exists(metas_path):
        raise FileNotFoundError(f"metas.json not found at: {metas_path}. Run embed.py to generate it.")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"extracted_chunks.csv not found at: {chunks_path}. Run extract_pdf.py to generate it.")

    emb = np.load(emb_path)
    metas = pd.read_json(metas_path)
    chunks = pd.read_csv(chunks_path)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb_norm = emb / (norm(emb, axis=1, keepdims=True) + 1e-12)

    print('Interactive ask. Enter a question (empty to quit).')
    while True:
        try:
            q = input('\nQuestion: ').strip()
        except EOFError:
            break
        if not q:
            break
        results = retrieve(q, emb, emb_norm, metas, chunks, model, k=args.k)
        print('\nTop passages:')
        for r in results:
            print(f"[{r['rank']}] (score={r['score']:.4f}) {r['source']} chunk={r['chunk_id']}\n{r['snippet']}\n")

        if not args.no_model:
            print('Querying model', args.model, 'for a label/prediction...')
            pred = call_model_predict(args.model, q)
            print('Model output:')
            print(json.dumps(pred, indent=2) if not isinstance(pred, str) else pred)

    print('Goodbye')

if __name__ == '__main__':
    main()
