# Neural Recipe Generation using Seq2Seq with Pointer-Generator and Coverage Mechanisms

- Designed and implemented multiple deep learning architectures for text generation using PyTorch, including baseline Seq2Seq, attention-based models, pointer-generator networks, and coverage mechanisms.
- Developed an extended vocabulary mechanism to handle out-of-vocabulary (OOV) tokens through dynamic copying from source sequences.
- Implemented beam search decoding with n-gram blocking and length normalization to improve generation quality and reduce repetition.
- Optimized training using gradient clipping, teacher forcing scheduling, and learning rate reduction on validation loss.
- Evaluated model performance using BLEU, METEOR, and BERTScore metrics, conducting comparative experiments across six model variants.
- Visualized and analyzed training convergence using loss tracking across architectures.

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
