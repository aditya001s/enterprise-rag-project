import argparse
import os
import json
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from gpu_utils import get_device, clear_gpu_memory

def retrieve_context(query, index_path, meta_path, chunks_path, embed_model, k=3):
    # Import here to avoid circular import
    from rag_retrieve import retrieve_with_faiss
    return retrieve_with_faiss(query, index_path, meta_path, chunks_path, embed_model, k)

def generate_response(query, context_chunks, generator_model_name='gpt2', device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    model = AutoModelForCausalLM.from_pretrained(generator_model_name).to(device)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare context
    context_text = "\n\n".join([chunk['full_text'] for chunk in context_chunks])
    
    # Create prompt
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            device=device if device == 'cpu' else None
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    answer_start = generated_text.find("Answer:") + len("Answer:")
    answer = generated_text[answer_start:].strip()
    
    # Clean up
    del model
    clear_gpu_memory()
    
    return answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True, help='User query')
    p.add_argument('--index', default='faiss_index.index', help='FAISS index path')
    p.add_argument('--meta', default='faiss_index_meta.json', help='Metadata JSON path')
    p.add_argument('--chunks', default='extracted_chunks.csv', help='Chunks CSV path')
    p.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')
    p.add_argument('--gen-model', default='gpt2', help='Generative model')
    p.add_argument('--k', type=int, default=3, help='Number of retrieved chunks')
    p.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = p.parse_args()
    
    device = get_device(use_gpu=args.gpu)
    
    # Load embedding model
    embed_model = SentenceTransformer(args.embed_model, device=str(device))
    
    # Retrieve context
    from rag_retrieve import retrieve_with_faiss
    context_chunks = retrieve_with_faiss(args.query, args.index, args.meta, args.chunks, embed_model, args.k, device=str(device))
    
    if not context_chunks:
        print("No relevant context found.")
        return
    
    # Generate response
    response = generate_response(args.query, context_chunks, args.gen_model, device=str(device))
    
    # Output
    output = {
        'query': args.query,
        'retrieved_chunks': context_chunks,
        'generated_response': response
    }
    
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()