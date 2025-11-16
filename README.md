Author: Himaja Arabati
Student ID: 700772489

This repository contains three implementations related to NLP sequence modeling and attention mechanisms.

1️⃣ Character-Level RNN Language Model

File: char_rnn_language_model.py

Description

Builds a simple character-level language model using RNN / LSTM / GRU.

Loads text, encodes characters, trains model to predict next character.

Generates text samples after training.

Features

Custom dataset preparation

Temperature-based sampling

Train + inference functions

Easy to extend for experiments

2️⃣ Bigram Language Model

File: bigram_language_model.py

Description

Implements a basic bigram model from scratch.

Learns probability of each character given the previous character.

Features

Frequency counting

Bigram probability matrix

Random text generation using learned distributions

Very fast + easy to understand baseline

3️⃣ Scaled Dot-Product Attention

File: scaled_dot_product_attention.py

Description

Implements the core attention formula used in Transformers:

Attention
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
T
	​

)V

Uses random Q, K, V tensors for testing.

Features

Computes attention weights

Computes attention outputs

Prints pre-scaling and post-scaling logits to show softmax stability

Clean, well-commented PyTorch implementation

📂 Project Structure
.
├── char_rnn_language_model.py
├── bigram_language_model.py
├── scaled_dot_product_attention.py
└── README.md

▶️ How to Run
python3 char_rnn_language_model.py
python3 bigram_language_model.py
python3 scaled_dot_product_attention.py
