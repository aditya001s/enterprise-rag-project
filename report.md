# Training Report

Dataset: `training/extracted_chunks.csv` (2,757 chunks from NIST SP 800-53 rev5)

Procedure:
- Created embeddings with `sentence-transformers/all-MiniLM-L6-v2` and clustered into 10 pseudo-labels.
- Trained three classifiers (RNN, LSTM, Self-Attention) on the pseudo-labeled dataset for 3 epochs each.

Results (final validation accuracy):
- RNN: 0.2138
- LSTM: 0.2319
- Self-Attention: 0.6486

Notes:
- Labels are synthetic (k-means clusters on semantic embeddings), so absolute accuracy is not directly meaningful for ground-truth tasks.
- Self-Attention model performed best on these clusters; consider longer training, hyperparameter tuning, or real labels for supervised tasks.

Artifacts:
- Extracted chunks: `training/extracted_chunks.csv`
- Embeddings: `training/embeddings.npy`
- Metas: `training/metas.json`
- Pseudo-labeled dataset: `training/labeled_dataset.csv`
- Saved models: `training/models/best_rnn.pt`, `training/models/best_lstm.pt`, `training/models/best_attn.pt`
- Evaluation script: `training/evaluate.py`

Next steps:
- If you have real labels, create `text,label` CSV and run `training/train.py` for supervised training.
- Export the best model for inference using `python training/evaluate.py --model attn --export-dir exported_attn`.
- Optionally install CUDA-enabled PyTorch and re-run training on GPU for faster results.
