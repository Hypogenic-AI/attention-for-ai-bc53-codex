# Attention in AI and the Internet: A Unified Bottleneck Analysis

## Project Overview
This project investigates the hypothesis that the **attention mechanism** is the fundamental principle common to both **Large Language Models (Generative AI)** and the **Internet Attention Economy (Recommendations, CTR)**. By modeling both domains with attention-based architectures, we demonstrate that a unified "attention bottleneck" explains performance across machine intelligence and human-computer digital markets.

## Key Findings
- **CTR Dominance**: Attention-based feature weighting (AttentionMLP) achieved **77.4% Accuracy** on Criteo CTR, outperforming a standard MLP (75.1%).
- **Sequential Equivalence**: Transformer-based sequential models match the performance of Recurrent Neural Networks (GRU) for user interaction sequences, achieving a **~0.25 Hit@10** on MovieLens.
- **Attention Focus (Entropy)**: Correct predictions are statistically linked to **lower attention entropy** (mean 1.41 vs 1.43), indicating that model "focus" is a key driver of success.

## How to Reproduce
1. **Environment Setup**:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. **Data Preparation**:
   ```bash
   uv run python src/data_prep.py
   ```
3. **Run Experiments**:
   ```bash
   export USER=researcher
   export TORCHINDUCTOR_CACHE_DIR=./.torchinductor_cache
   uv run python src/exp1_movielens_seq.py
   uv run python src/exp2_criteo_ctr.py
   ```
4. **Run Analysis**:
   ```bash
   uv run python src/analysis.py
   ```

## File Structure
- `src/`: Core experimental scripts.
- `datasets/`: MovieLens and Criteo data (after running `data_prep.py`).
- `results/`: Model checkpoints and JSON results.
- `results/plots/`: Visualizations of attention entropy and weights.
- `REPORT.md`: Full research report.

## Full Report
For detailed methodology, metrics, and analysis, please refer to [REPORT.md](./REPORT.md).
