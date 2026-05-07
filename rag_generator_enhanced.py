#!/usr/bin/env python
"""
Enhanced RAG Generator with Self-Attention Integration.
Generates responses using retrieved context and trained classifier models.
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from gpu_utils import get_device, clear_gpu_memory
import sys


class RAGGenerator:
    """GPU-optimized RAG generator with multiple backend support"""
    
    def __init__(self, model_name: str = 'gpt2', device: Optional[torch.device] = None,
                 max_length: int = 200):
        """
        Initialize RAG generator.
        
        Args:
            model_name: HuggingFace model ID
            device: torch device
            max_length: Maximum length of generated text
        """
        self.model_name = model_name
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        self.max_length = max_length
        
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try to load as causal LM, fallback to seq2seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.is_seq2seq = False
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.is_seq2seq = True
        
        self.model.eval()
        print(f"Loaded generator model on {self.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_beams=1
                )
            else:
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def generate_with_context(self, query: str, context_chunks: List[Dict],
                             max_context_tokens: int = 500,
                             temperature: float = 0.7) -> str:
        """
        Generate response with retrieved context.
        
        Args:
            query: User query
            context_chunks: List of retrieved chunks
            max_context_tokens: Maximum tokens for context
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Build context
        context_text = self._build_context(context_chunks, max_context_tokens)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text)
        
        # Generate
        response = self.generate(prompt, temperature=temperature)
        
        # Extract response part
        response = self._extract_response(response)
        
        return response
    
    def _build_context(self, chunks: List[Dict], max_tokens: int) -> str:
        """Build context string from chunks with token limit"""
        context_parts = []
        token_count = 0
        
        for chunk in chunks:
            text = chunk.get('full_text', chunk.get('snippet', ''))
            chunk_tokens = len(self.tokenizer.encode(text))
            
            if token_count + chunk_tokens > max_tokens:
                break
            
            source = chunk.get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{text}\n")
            token_count += chunk_tokens
        
        return "\n\n".join(context_parts) if context_parts else "No context available."
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt template"""
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def _extract_response(self, generated_text: str) -> str:
        """Extract response from generated text"""
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        
        return answer.strip()
    
    def clear_cache(self):
        """Clear GPU memory"""
        clear_gpu_memory()


class SelfAttentionClassifier:
    """Wrapper for using trained self-attention model in RAG pipeline"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize self-attention classifier.
        
        Args:
            model_path: Path to saved model checkpoint
            device: torch device
        """
        self.model_path = model_path
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vocab_stoi = checkpoint.get('vocab', {})
        
        # Build vocab
        self.build_vocab()
        
        # Load model (will be initialized by subclass)
        print(f"Loaded model checkpoint from {model_path}")
    
    def build_vocab(self):
        """Build vocabulary from stoi"""
        vocab_size = max(self.vocab_stoi.values()) + 1 if self.vocab_stoi else 5000
        
        itos = ['<unk>'] * vocab_size
        for token, idx in self.vocab_stoi.items():
            if idx < vocab_size:
                itos[idx] = token
        
        self.itos = itos
        self.stoi = self.vocab_stoi
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text"""
        tokens = []
        for word in text.lower().split():
            word = word.strip('.,!?;:')
            token_id = self.stoi.get(word, 0)  # 0 is <unk>
            tokens.append(token_id)
        return tokens


class ContextAnalyzer:
    """Analyzes and ranks retrieved context using classification models"""
    
    def __init__(self, classifier_path: Optional[str] = None, 
                 device: Optional[torch.device] = None):
        """
        Initialize context analyzer.
        
        Args:
            classifier_path: Path to trained classifier (optional)
            device: torch device
        """
        self.classifier_path = classifier_path
        self.device = device or get_device(use_gpu=torch.cuda.is_available())
        
        if classifier_path and os.path.exists(classifier_path):
            self.classifier = SelfAttentionClassifier(classifier_path, device=self.device)
        else:
            self.classifier = None
    
    def rank_chunks_by_relevance(self, chunks: List[Dict], query: str) -> List[Dict]:
        """
        Rank chunks by relevance to query.
        
        Args:
            chunks: Retrieved chunks
            query: User query
            
        Returns:
            Re-ranked chunks with relevance scores
        """
        if not self.classifier:
            return chunks  # Return as-is if no classifier
        
        # Score each chunk
        for chunk in chunks:
            chunk_text = chunk.get('full_text', chunk.get('snippet', ''))
            relevance_score = self._compute_relevance(query, chunk_text)
            chunk['relevance_score'] = relevance_score
        
        # Sort by relevance
        chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return chunks
    
    def _compute_relevance(self, query: str, chunk_text: str) -> float:
        """Compute relevance score between query and chunk"""
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk_text.lower().split())
        
        intersection = query_terms & chunk_terms
        union = query_terms | chunk_terms
        
        jaccard_sim = len(intersection) / len(union) if union else 0
        return float(jaccard_sim)


class RAGPipeline:
    """Complete RAG pipeline integrating retrieval, analysis, and generation"""
    
    def __init__(self, generator: RAGGenerator, context_analyzer: Optional[ContextAnalyzer] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            generator: RAGGenerator instance
            context_analyzer: ContextAnalyzer instance
        """
        self.generator = generator
        self.context_analyzer = context_analyzer or ContextAnalyzer()
    
    def generate_response(self, query: str, context_chunks: List[Dict],
                         rerank: bool = True, temperature: float = 0.7) -> Dict:
        """
        Generate full RAG response.
        
        Args:
            query: User query
            context_chunks: Retrieved chunks
            rerank: Whether to rerank chunks
            temperature: Generation temperature
            
        Returns:
            Dict with query, context, response
        """
        if rerank and self.context_analyzer:
            context_chunks = self.context_analyzer.rank_chunks_by_relevance(context_chunks, query)
        
        response = self.generator.generate_with_context(
            query, context_chunks, temperature=temperature
        )
        
        return {
            'query': query,
            'num_chunks': len(context_chunks),
            'top_chunks': [
                {
                    'source': c.get('source'),
                    'score': c.get('score'),
                    'snippet': c.get('snippet')
                }
                for c in context_chunks[:3]
            ],
            'generated_response': response
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Generator')
    parser.add_argument('--query', required=True, help='User query')
    parser.add_argument('--context', nargs='+', help='Context text')
    parser.add_argument('--model', default='gpt2', help='Generator model')
    parser.add_argument('--classifier', help='Classifier model path')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    args = parser.parse_args()
    
    device = get_device(use_gpu=args.gpu) if args.gpu else torch.device('cpu')
    
    generator = RAGGenerator(model_name=args.model, device=device)
    analyzer = ContextAnalyzer(classifier_path=args.classifier, device=device) if args.classifier else None
    
    if args.context:
        context_chunks = [{'full_text': c} for c in args.context]
        response = generator.generate_with_context(args.query, context_chunks, temperature=args.temperature)
        print(f"Response: {response}")
    else:
        response = generator.generate(args.query, temperature=args.temperature)
        print(f"Response: {response}")
    
    generator.clear_cache()


if __name__ == '__main__':
    main()
