import argparse
import os
import json
import pandas as pd
from pypdf import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

def read_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                text.append(txt)
    return "\n".join(text)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks

def main(pdf_path, out_csv):
    txt = read_pdf(pdf_path)
    chunks = chunk_text(txt)
    rows = []
    for i, c in enumerate(chunks):
        rows.append({"text": c, "source": os.path.abspath(pdf_path), "chunk_id": i})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} chunks to {out_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', required=True, help='Path to PDF')
    p.add_argument('--out', default='extracted_chunks.csv', help='Output CSV path')
    args = p.parse_args()
    main(args.pdf, args.out)
