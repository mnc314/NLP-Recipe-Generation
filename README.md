# Neural Recipe Generation using Seq2Seq with Pointer-Generator and Coverage Mechanisms

This project implements multiple neural sequence-to-sequence architectures for text generation, progressively enhancing model capability through advanced decoding and attention mechanisms.

## Implemented Models
- Base Seq2Seq model
- Attention-based Seq2Seq
- Pointer-Generator Network
- Coverage Mechanism model
- Beam Search decoding
- OOV handling with extended vocabulary

## Key Features
- Dynamic vocabulary extension for rare tokens
- Coverage penalty to reduce repetition
- Beam search decoding with n-gram blocking
- Teacher forcing scheduling
- Gradient clipping
- Learning rate scheduling
- Comparative loss visualization across models

## Evaluation Metrics
- BLEU
- METEOR
- BERTScore

## Technologies
- Python
- PyTorch
- NLTK
- Matplotlib
- Pandas
