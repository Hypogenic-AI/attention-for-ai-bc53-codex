# Resources Catalog

## Summary
This catalog contains the resources gathered to investigate the hypothesis that attention mechanisms underlie the core mechanisms of Generative AI and the Internet.

## Papers
| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| An attention economy model of co-evolution | Chujyo et al. | 2026 | an_attention_economy_model_of_.pdf | Mathematical framework for quality vs selectivity. |
| Push and Pull: Attentional Agency | Wojtowicz et al. | 2024 | push_and_pull_a_framework_for_.pdf | Formalizing user vs advocate value in platforms. |
| RecGRELA: Long-term Recommendation | Hu et al. | 2025 | rec_grela.pdf | Technical optimization of linear attention. |
| Stability of Personalised Markets | Zhu et al. | 2023 | stability_and_efficiency_of_pe.pdf | Connection between influence and optimization. |
| On Online Attention Dynamics | Castaldo et al. | 2022 | on_online_attention_dynamics.pdf | Review of collective attention. |

## Datasets
| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MovieLens Ratings | HuggingFace | 100K | Recommendation | datasets/movielens/ | Classic preference dataset. |
| Criteo x1 | HuggingFace | 1M+ | CTR Prediction | datasets/criteo/ | Standard for attention economy in ads. |

## Code Repositories
| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RecGRELA | github.com/TBI805/RecGRELA | Linear Attention Impl | code/rec_grela/ | Gated Rotary-Enhanced Linear Attention. |
| PANTHER | github.com/WeChatPay-Pretraining/PANTHER | Generative Pretraining | code/panther/ | Sequential behavior modeling at scale. |
| FuXi-Linear | github.com/USTC-StarTeam/fuxi-linear | Time-aware Attention | code/fuxi_linear/ | Long-term sequential recommendation. |

## Recommendations for Experiment Design
1. **Bridge Theory and Practice**: Use the mathematical parameters from Chujyo et al. (2026) (like sigma attention capacity) as constraints in a recommendation model (RecGRELA).
2. **Generative Hypothesis**: Test if "Generative Pretraining" on user sequences (PANTHER) outperforms traditional discriminative models in capturing "Attention."
3. **Agency Metric**: Implement the Push/Pull metrics to evaluate if more sophisticated attention mechanisms increase "user agency" or "platform push."
