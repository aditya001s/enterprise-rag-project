import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

def topk_for_query(query, emb, emb_norm, metas, chunks, model, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    qn = q_emb / (norm(q_emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_norm @ qn[0]
    idxs = sims.argsort()[-k:][::-1]
    out = []
    for rank, idx in enumerate(idxs, 1):
        meta = metas.iloc[idx]
        text = chunks.iloc[idx]['text'] if 'text' in chunks.columns else ''
        snippet = text.replace('\n',' ')[:800]
        out.append({'rank': rank, 'score': float(sims[idx]), 'source': meta.get('source'), 'chunk_id': int(meta.get('chunk_id')), 'snippet': snippet})
    return out

def main():
    emb = np.load('training/embeddings.npy')
    metas = pd.read_json('training/metas.json')
    chunks = pd.read_csv('training/extracted_chunks.csv')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb_norm = emb / (norm(emb, axis=1, keepdims=True) + 1e-12)

    queries = [
        "What are access control requirements?",
        "How should organizations handle incident response?",
        "What privacy controls support data minimization?"
    ]

    for q in queries:
        print('\n=== QUERY:', q, '===')
        results = topk_for_query(q, emb, emb_norm, metas, chunks, model, k=3)
        for r in results:
            print(f"Rank {r['rank']} score={r['score']:.4f} source={r['source']} chunk_id={r['chunk_id']}")
            print(r['snippet'])

if __name__ == '__main__':
    main()

