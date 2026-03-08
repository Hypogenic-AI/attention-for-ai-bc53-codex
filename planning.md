# Research Plan: Attention as a Cross-Domain Bottleneck in AI and Internet Systems

## Motivation & Novelty Assessment

### Why This Research Matters
Attention is a scarce resource in both machine intelligence and human-computer systems, so testing whether a shared attention principle predicts performance across domains has practical value for model design, ranking systems, and product optimization. If validated, this supports a unified framing that connects generative AI architecture choices with Internet traffic allocation behavior. That can improve how we design, evaluate, and govern attention-driven systems.

### Gap in Existing Work
From `literature_review.md`, prior work demonstrates attention effectiveness in separate silos (transformers, diffusion conditioning, recommender/CTR), but does not empirically test a single cross-domain claim with a common experimental protocol and comparable metrics. In particular, evidence linking internal model attention behavior to external attention-allocation outcomes is limited.

### Our Novel Contribution
We run a unified empirical study across two local datasets representing (1) AI language understanding and (2) Internet attention outcomes, comparing attention vs non-attention baselines under consistent statistical analysis. We additionally test whether attention concentration (entropy-based) is associated with better predictive performance, providing a measurable bridge between mechanism and outcome.

### Experiment Justification
- Experiment 1: AG News model comparison (non-attention vs transformer attention). Needed to test whether attention-centric models outperform non-attention baselines in a controlled NLP task.
- Experiment 2: Online News Popularity model comparison (MLP vs feature self-attention). Needed to test whether attention-style weighting improves prediction of user attention proxies in Internet content outcomes.
- Experiment 3: Attention concentration analysis. Needed to test whether the *quality/distribution* of learned attention links to prediction correctness, supporting mechanism-level interpretation beyond raw score improvements.

## Research Question
Does attention function as a unifying bottleneck mechanism across AI and Internet systems, such that attention-based models outperform non-attention baselines and attention concentration patterns are associated with stronger outcomes?

## Background and Motivation
The hypothesis proposes that transformer self-attention, generative AI conditioning, and Internet ranking/CTR systems share one mathematical logic: selective weighting over limited information budget. Literature in this workspace supports each component independently (Vaswani, DIN, SASRec, BERT4Rec, LDM), but lacks an integrated experimental test with shared evaluation logic.

## Hypothesis Decomposition
- H1 (AI task): On AG News, an attention-based model (DistilBERT) outperforms a non-attention baseline (TF-IDF + Logistic Regression) on macro-F1 and accuracy.
- H2 (Internet proxy task): On Online News Popularity, a feature self-attention network outperforms a similarly sized MLP on AUC, F1, and NDCG@10 for high-share prediction.
- H3 (Mechanism link): Lower attention entropy (more concentrated attention) is associated with improved correctness/confidence in both domains.

Independent variables:
- Model family (`non_attention`, `attention`)
- Dataset/task (`ag_news`, `online_news_popularity`)
- Random seed (42, 43, 44)

Dependent variables:
- AG News: accuracy, macro-F1
- Online News: ROC-AUC, F1, NDCG@10
- Mechanism: attention entropy, correlation with correctness/confidence

Alternative explanations:
- Parameter count and pretraining, not attention per se
- Feature engineering effects on tabular data
- Class imbalance artifacts

Mitigations:
- Report model sizes and training settings
- Use standardized preprocessing and consistent splits
- Use multi-seed averages with significance tests

## Proposed Methodology

### Approach
A two-domain comparative benchmark using pre-gathered datasets and lightweight reproducible models. We prioritize feasible, replicable CPU/GPU execution in one session and use statistical tests plus effect sizes.

### Experimental Steps
1. Environment and reproducibility setup (venv, dependency lock, seeds, logging).
2. Data loading and quality checks for both datasets.
3. AG News baseline training: TF-IDF + Logistic Regression.
4. AG News attention model: DistilBERT fine-tuning (small subset if compute constrained).
5. Online News preprocessing and binary target construction (`shares` above median).
6. Online News baseline: MLP classifier.
7. Online News attention model: feature self-attention classifier.
8. Multi-seed evaluation and bootstrap confidence intervals.
9. Statistical tests (paired t-test/Wilcoxon based on normality checks).
10. Attention entropy analysis and failure-case analysis.

### Baselines
- AG News baseline: TF-IDF + Logistic Regression (strong classic non-attention baseline).
- Online News baseline: MLP classifier (capacity-matched non-attention neural baseline).

### Evaluation Metrics
- Accuracy and macro-F1 for AG News (class-balanced performance visibility).
- ROC-AUC, F1, and NDCG@10 for Online News (classification + ranking relevance to attention allocation).
- Attention entropy and Pearson/Spearman correlations with correctness/confidence.

### Statistical Analysis Plan
- Significance threshold: alpha = 0.05.
- For model comparisons across seeds: paired test per metric.
- Normality check via Shapiro-Wilk on paired differences.
- If normal: paired t-test; if non-normal: Wilcoxon signed-rank.
- Effect sizes: Cohen's d (paired) and Cliff's delta (non-parametric robustness where useful).
- 95% bootstrap confidence intervals (1,000 resamples) for key metrics.

## Expected Outcomes
Support for hypothesis:
- Attention models outperform non-attention baselines on both domains.
- Attention entropy shows consistent relationship with better outcomes.

Refutation/partial support:
- Gains only in one domain, or no clear mechanism link in entropy analysis.

## Timeline and Milestones
- Phase A (20 min): setup, data validation, EDA
- Phase B (40 min): implement/train AG News models
- Phase C (40 min): implement/train Online News models
- Phase D (25 min): statistical analysis + plots
- Phase E (25 min): REPORT.md + README.md + validation
- Buffer (30%): debugging and reruns

## Potential Challenges
- No GPU available: reduce DistilBERT sample size and epochs.
- Runtime constraints: use stratified subsampling while preserving test integrity.
- Attention extraction complexity: rely on model outputs with attention tensors and standardized entropy computation.
- Confounding from pretraining: discuss as limitation and include parameter/context transparency.

## Success Criteria
- End-to-end reproducible pipeline runs from command line.
- `results/` contains raw metrics, plots, and per-seed outputs.
- REPORT documents statistical tests and mechanism analysis.
- At least one cross-domain positive result and one clearly discussed limitation.
