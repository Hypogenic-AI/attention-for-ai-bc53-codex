# Literature Review: Attention is All You Need for the Internet Field

## Research Area Overview
The research explores the hypothesis that the attention mechanism, originally popularized by Transformers in Large Language Models (LLMs), is a fundamental principle not only for artificial intelligence but also for the entire Generative AI and Internet field. This field is characterized by the "Attention Economy," where digital products compete for the limited cognitive resources of users.

## Key Papers

### 1. An attention economy model of co-evolution between content quality and audience selectivity (2026)
- **Authors**: M. Chujyo, I. Okada, H. Yamamoto, D. Lim, F. Toriumi
- **Key Contribution**: Develops a minimal mathematical framework (two-population evolutionary game) to explain how content quality and audience selectivity co-evolve under limited attention capacity.
- **Methodology**: Replicator dynamics, mean-field approximation.
- **Relevance**: Provides the mathematical bridge between "attention capacity" and the survival of high-quality content in digital ecosystems.

### 2. Push and Pull: A Framework for Measuring Attentional Agency on Digital Platforms (2024)
- **Authors**: Z. Wojtowicz, S. Jain, N. Vincent
- **Key Contribution**: Defines "attentional agency" and a formal framework for measuring it. Distinguishes between "Pull" (value to user) and "Push" (value to advocates/advertisers).
- **Relevance**: Connects the technical mediation of information (by foundation models and ranking systems) to the strategic objectives of platforms.

### 3. Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation (2025)
- **Authors**: J. Hu, W. Zhou, et al.
- **Key Contribution**: Introduces RecGRELA, a model using rotary-enhanced linear attention for processing long-term user behavior sequences.
- **Methodology**: Linear attention, rotary position embeddings, rank modulation.
- **Relevance**: Demonstrates how technical attention mechanisms are optimized for the core "Internet" task of sequential recommendation.

### 4. PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling (2025)
- **Authors**: WeChat Pay Team
- **Key Contribution**: Applies generative pretraining (similar to LLMs) to industrial-scale user behavior data (credit card transactions).
- **Relevance**: Proves that the "Attention is All You Need" paradigm (generative pretraining + attention) is directly applicable to non-language internet datasets.

### 5. On Online Attention Dynamics (2022)
- **Authors**: M. Castaldo, P. Frasca, T. Venturini
- **Key Contribution**: A review of collective attention dynamics and the need for mathematical modeling of content dissemination.
- **Relevance**: Establishes the historical context of attention research on the internet.

## Common Methodologies
- **Attention Mechanisms**: Transformer-based attention (scaled dot-product), Linear Attention (for efficiency), Gated Attention.
- **Game Theory**: Evolutionary games to model provider-consumer feedback loops.
- **Sequential Modeling**: Treating user internet behavior (clicks, purchases, views) as a sequence similar to tokens in a sentence.

## Standard Baselines
- **Recommendation**: SASRec, Bert4Rec, GRU4Rec.
- **Ranking**: LambdaMART, Deep Interest Network (DIN).
- **Attention Models**: Standard Transformer, Mamba (as a competitor for long sequences).

## Evaluation Metrics
- **Technical**: NDCG, Hit Rate, AUC, Click-Through Rate (CTR).
- **Economic**: Attention Capacity (sigma), Attentional Agency (Push/Pull scores).

## Datasets in the Literature
- **MovieLens (ML-1M, ML-20M)**: Standard for recommendation.
- **Criteo**: Standard for CTR prediction.
- **Baidu-ULTR**: Industrial scale user behavior logs.
- **Tmall / Amazon**: E-commerce sequential behavior.

## Recommendations for Our Experiment
1. **Primary Dataset**: Use MovieLens for baseline recommendation and Criteo for CTR analysis.
2. **Model to Adapt**: Use the RecGRELA or PANTHER architecture as a technical implementation of the attention mechanism.
3. **Simulation**: Implement the Chujyo et al. (2026) model to simulate how different "attention mechanisms" (e.g., higher discriminability) affect the co-evolution of content quality.
