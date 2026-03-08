import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_from_disk
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Prevent getpass/pwd lookup failures in containerized UID-only environments.
os.environ.setdefault("USER", "codex")
os.environ.setdefault("LOGNAME", "codex")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.cwd() / ".torchinductor_cache"))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_OUTPUT_DIR = RESULTS_DIR / "model_outputs"
LOG_DIR = ROOT / "logs"


@dataclass
class Config:
    seeds: List[int]
    ag_model_name: str = "distilbert-base-uncased"
    ag_max_len: int = 128
    ag_train_subset: int = 6000
    ag_val_subset: int = 1500
    ag_batch_size: int = 64
    ag_epochs: int = 1
    ag_lr: float = 2e-5
    ag_weight_decay: float = 0.01
    online_batch_size: int = 128
    online_epochs: int = 20
    online_lr: float = 2e-3
    online_d_model: int = 64
    online_n_heads: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "run_experiments.log"),
            logging.StreamHandler(),
        ],
    )


def env_info() -> Dict:
    gpu_info = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_info.append(
                {
                    "index": idx,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
    }


def normalized_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -(probs * np.log(probs)).sum(axis=axis)
    n = probs.shape[axis]
    return ent / np.log(n)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_agnews_baseline(seed: int) -> Dict:
    set_seed(seed)
    ds = load_from_disk(str(ROOT / "datasets" / "ag_news"))
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    X_train, _, y_train, _ = train_test_split(
        train_texts,
        train_labels,
        train_size=6000,
        stratify=train_labels,
        random_state=seed,
    )

    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        C=4.0,
        max_iter=1200,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "macro_f1": float(f1_score(test_labels, preds, average="macro")),
    }



def run_agnews_attention(seed: int, cfg: Config, device: torch.device) -> Tuple[Dict, Dict]:
    set_seed(seed)
    ds = load_from_disk(str(ROOT / "datasets" / "ag_news"))

    train_df = pd.DataFrame({"text": ds["train"]["text"], "label": ds["train"]["label"]})
    test_df = pd.DataFrame({"text": ds["test"]["text"], "label": ds["test"]["label"]})

    tr_df, val_df = train_test_split(
        train_df,
        test_size=cfg.ag_val_subset,
        stratify=train_df["label"],
        random_state=seed,
    )
    tr_df, _ = train_test_split(
        tr_df,
        train_size=cfg.ag_train_subset,
        stratify=tr_df["label"],
        random_state=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.ag_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.ag_model_name,
        num_labels=4,
        attn_implementation="eager",
    ).to(device)

    # Freeze backbone to make multi-seed runs feasible in one session.
    for name, param in model.named_parameters():
        if "classifier" not in name and "pre_classifier" not in name:
            param.requires_grad = False

    train_ds = TextDataset(tr_df["text"].tolist(), tr_df["label"].tolist(), tokenizer, cfg.ag_max_len)
    val_ds = TextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, cfg.ag_max_len)
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, cfg.ag_max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.ag_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.ag_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.ag_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.ag_lr,
        weight_decay=cfg.ag_weight_decay,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    model.train()
    for _ in range(cfg.ag_epochs):
        for batch in tqdm(train_loader, desc=f"AG News train seed={seed}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    def eval_loader(loader, with_attn=False):
        model.eval()
        y_true, y_pred, conf = [], [], []
        entropies = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_attentions=with_attn,
                )
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                y_true.extend(batch["labels"].cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                conf.extend(probs.max(dim=-1).values.cpu().numpy().tolist())

                if with_attn:
                    # last layer attention: [B, heads, T, T]
                    attn = out.attentions[-1].detach().cpu().numpy()
                    # average over heads and query positions -> distribution over key tokens
                    p = attn.mean(axis=1).mean(axis=1)
                    entropies.extend(normalized_entropy(p, axis=-1).tolist())

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        extra = {
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": conf,
            "attention_entropy": entropies,
        }
        return metrics, extra

    val_metrics, _ = eval_loader(val_loader, with_attn=False)
    test_metrics, test_extra = eval_loader(test_loader, with_attn=True)
    test_metrics["val_accuracy"] = val_metrics["accuracy"]

    return test_metrics, test_extra


class FeatureSelfAttentionNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int):
        super().__init__()
        self.n_features = n_features
        self.feature_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.cls = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        attn_out, attn_w = self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        pooled = self.ffn(attn_out.mean(dim=1))
        logits = self.cls(pooled).squeeze(-1)
        return logits, attn_w


def load_online_news() -> pd.DataFrame:
    path = ROOT / "datasets" / "online_news_popularity" / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_online_baseline(seed: int, split: Dict) -> Dict:
    set_seed(seed)
    X_train, X_test, y_train, y_test = split["X_train"], split["X_test"], split["y_train"], split["y_test"]

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=15,
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "ndcg10": float(ndcg_score([y_test], [probs], k=10)),
    }


def run_online_attention(seed: int, split: Dict, cfg: Config, device: torch.device) -> Tuple[Dict, Dict]:
    set_seed(seed)
    X_train = torch.tensor(split["X_train"], dtype=torch.float32)
    y_train = torch.tensor(split["y_train"], dtype=torch.float32)
    X_val = torch.tensor(split["X_val"], dtype=torch.float32)
    y_val = torch.tensor(split["y_val"], dtype=torch.float32)
    X_test = torch.tensor(split["X_test"], dtype=torch.float32)
    y_test = torch.tensor(split["y_test"], dtype=torch.float32)

    model = FeatureSelfAttentionNet(
        n_features=X_train.shape[1],
        d_model=cfg.online_d_model,
        n_heads=cfg.online_n_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.online_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=device.type == "cuda")

    train_loader = DataLoader(
        list(zip(X_train, y_train)),
        batch_size=cfg.online_batch_size,
        shuffle=True,
    )

    best_val_auc = -1.0
    best_state = None

    for _ in range(cfg.online_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits, _ = model(xb, need_weights=False)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val.to(device), need_weights=False)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits, attn = model(X_test.to(device), need_weights=True)
        probs = torch.sigmoid(test_logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        # attn: [B, heads, F, F]
        attn_np = attn.detach().cpu().numpy()
        p = attn_np.mean(axis=1).mean(axis=1)
        entropy = normalized_entropy(p, axis=-1)

    metrics = {
        "auc": float(roc_auc_score(y_test.numpy(), probs)),
        "f1": float(f1_score(y_test.numpy(), preds)),
        "ndcg10": float(ndcg_score([y_test.numpy()], [probs], k=10)),
    }
    extra = {
        "y_true": y_test.numpy().tolist(),
        "y_pred": preds.tolist(),
        "score": probs.tolist(),
        "attention_entropy": entropy.tolist(),
    }
    return metrics, extra


def paired_stats(base: List[float], attn: List[float]) -> Dict:
    base = np.array(base, dtype=float)
    attn = np.array(attn, dtype=float)
    diff = attn - base
    shapiro_p = float(stats.shapiro(diff).pvalue)

    if shapiro_p > 0.05:
        test_name = "paired_t_test"
        stat, p = stats.ttest_rel(attn, base)
    else:
        test_name = "wilcoxon_signed_rank"
        stat, p = stats.wilcoxon(attn, base)

    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))

    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(diff), len(diff))
        boots.append(float(diff[idx].mean()))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    return {
        "difference_mean": float(diff.mean()),
        "difference_std": float(diff.std(ddof=1)),
        "normality_shapiro_p": shapiro_p,
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "cohens_d_paired": d,
        "ci95": [float(ci_low), float(ci_high)],
    }


def make_plots(metrics: Dict, ag_entropy: Dict, online_entropy: Dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # AG News metrics
    ag_rows = []
    for model_name, vals in metrics["ag_news"].items():
        for seed, res in vals.items():
            ag_rows.append({"model": model_name, "seed": seed, "accuracy": res["accuracy"], "macro_f1": res["macro_f1"]})
    ag_df = pd.DataFrame(ag_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ag_df.melt(id_vars=["model", "seed"], value_vars=["accuracy", "macro_f1"], var_name="metric"), x="metric", y="value", hue="model", errorbar="sd", ax=ax)
    ax.set_title("AG News: Non-Attention vs Attention")
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "agnews_metric_comparison.png", dpi=180)
    plt.close(fig)

    # Online metrics
    on_rows = []
    for model_name, vals in metrics["online_news"].items():
        for seed, res in vals.items():
            on_rows.append({"model": model_name, "seed": seed, "auc": res["auc"], "f1": res["f1"], "ndcg10": res["ndcg10"]})
    on_df = pd.DataFrame(on_rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=on_df.melt(id_vars=["model", "seed"], value_vars=["auc", "f1", "ndcg10"], var_name="metric"), x="metric", y="value", hue="model", errorbar="sd", ax=ax)
    ax.set_title("Online News: Non-Attention vs Attention")
    ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "online_metric_comparison.png", dpi=180)
    plt.close(fig)

    # Entropy vs correctness
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(x=ag_entropy["correct"], y=ag_entropy["entropy"], ax=axes[0])
    axes[0].set_title("AG News Attention Entropy vs Correctness")
    axes[0].set_xlabel("Correct (0/1)")
    axes[0].set_ylabel("Normalized Attention Entropy")

    sns.boxplot(x=online_entropy["correct"], y=online_entropy["entropy"], ax=axes[1])
    axes[1].set_title("Online News Attention Entropy vs Correctness")
    axes[1].set_xlabel("Correct (0/1)")
    axes[1].set_ylabel("Normalized Attention Entropy")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "attention_entropy_correctness.png", dpi=180)
    plt.close(fig)


def summarize(vals: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    keys = next(iter(vals.values())).keys()
    for k in keys:
        arr = np.array([vals[s][k] for s in vals], dtype=float)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
    return out


def main():
    setup_logging()
    cfg = Config(seeds=[42, 43, 44])

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    info = env_info()
    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(info, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    logging.info("Config: %s", cfg)

    # Data quality checks for Online News
    df_online = load_online_news()
    feature_cols = [c for c in df_online.columns if c not in ["url", "timedelta", "shares"]]

    missing_pct = float((df_online[feature_cols].isna().mean().mean()) * 100)
    duplicate_rows = int(df_online.duplicated().sum())
    z = np.abs(stats.zscore(df_online[feature_cols], nan_policy="omit"))
    outlier_rows = int((z > 3).any(axis=1).sum())

    quality = {
        "online_news": {
            "rows": int(df_online.shape[0]),
            "columns": int(df_online.shape[1]),
            "missing_pct": missing_pct,
            "duplicates": duplicate_rows,
            "outlier_rows_z_gt_3": outlier_rows,
            "shares_median": float(df_online["shares"].median()),
        }
    }

    # Build split once and reuse across seeds for fair model comparison.
    y_bin = (df_online["shares"] >= df_online["shares"].median()).astype(int).values
    X = df_online[feature_cols].values.astype(np.float32)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y_bin,
        test_size=0.3,
        random_state=42,
        stratify=y_bin,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=42,
        stratify=y_tmp,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    split = {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    metrics = {
        "ag_news": {"baseline": {}, "attention": {}},
        "online_news": {"baseline": {}, "attention": {}},
    }

    ag_entropy_agg = {"entropy": [], "correct": [], "confidence": []}
    on_entropy_agg = {"entropy": [], "correct": [], "confidence": []}

    t0 = time.time()

    for seed in cfg.seeds:
        logging.info("Running seed %d", seed)
        # AG News
        ag_base = run_agnews_baseline(seed)
        metrics["ag_news"]["baseline"][seed] = ag_base

        ag_attn, ag_extra = run_agnews_attention(seed, cfg, device)
        metrics["ag_news"]["attention"][seed] = ag_attn

        ag_correct = (np.array(ag_extra["y_true"]) == np.array(ag_extra["y_pred"])).astype(int)
        ag_entropy_agg["entropy"].extend(ag_extra["attention_entropy"])
        ag_entropy_agg["correct"].extend(ag_correct.tolist())
        ag_entropy_agg["confidence"].extend(ag_extra["confidence"])

        # Online News
        on_base = run_online_baseline(seed, split)
        metrics["online_news"]["baseline"][seed] = on_base

        on_attn, on_extra = run_online_attention(seed, split, cfg, device)
        metrics["online_news"]["attention"][seed] = on_attn

        on_correct = (np.array(on_extra["y_true"]) == np.array(on_extra["y_pred"])).astype(int)
        on_entropy_agg["entropy"].extend(on_extra["attention_entropy"])
        on_entropy_agg["correct"].extend(on_correct.tolist())
        on_entropy_agg["confidence"].extend(on_extra["score"])

    runtime_min = (time.time() - t0) / 60.0

    stats_results = {
        "ag_news": {},
        "online_news": {},
    }

    for metric in ["accuracy", "macro_f1"]:
        base = [metrics["ag_news"]["baseline"][s][metric] for s in cfg.seeds]
        attn = [metrics["ag_news"]["attention"][s][metric] for s in cfg.seeds]
        stats_results["ag_news"][metric] = paired_stats(base, attn)

    for metric in ["auc", "f1", "ndcg10"]:
        base = [metrics["online_news"]["baseline"][s][metric] for s in cfg.seeds]
        attn = [metrics["online_news"]["attention"][s][metric] for s in cfg.seeds]
        stats_results["online_news"][metric] = paired_stats(base, attn)

    # Attention concentration relationships
    ag_entropy = np.array(ag_entropy_agg["entropy"])
    ag_correct = np.array(ag_entropy_agg["correct"])
    ag_conf = np.array(ag_entropy_agg["confidence"])

    on_entropy = np.array(on_entropy_agg["entropy"])
    on_correct = np.array(on_entropy_agg["correct"])
    on_conf = np.array(on_entropy_agg["confidence"])

    entropy_analysis = {
        "ag_news": {
            "entropy_mean": float(ag_entropy.mean()),
            "entropy_std": float(ag_entropy.std(ddof=1)),
            "spearman_entropy_correct": float(stats.spearmanr(ag_entropy, ag_correct).correlation),
            "spearman_entropy_confidence": float(stats.spearmanr(ag_entropy, ag_conf).correlation),
            "mean_entropy_correct": float(ag_entropy[ag_correct == 1].mean()),
            "mean_entropy_wrong": float(ag_entropy[ag_correct == 0].mean()),
        },
        "online_news": {
            "entropy_mean": float(on_entropy.mean()),
            "entropy_std": float(on_entropy.std(ddof=1)),
            "spearman_entropy_correct": float(stats.spearmanr(on_entropy, on_correct).correlation),
            "spearman_entropy_score": float(stats.spearmanr(on_entropy, on_conf).correlation),
            "mean_entropy_correct": float(on_entropy[on_correct == 1].mean()),
            "mean_entropy_wrong": float(on_entropy[on_correct == 0].mean()),
        },
    }

    summary = {
        "environment": info,
        "runtime_minutes": runtime_min,
        "config": cfg.__dict__,
        "data_quality": quality,
        "split_sizes": {
            "online_news": {
                "train": int(len(y_train)),
                "val": int(len(y_val)),
                "test": int(len(y_test)),
                "positive_rate_train": float(np.mean(y_train)),
                "positive_rate_test": float(np.mean(y_test)),
            }
        },
        "metrics": metrics,
        "aggregate": {
            "ag_news": {
                "baseline": summarize(metrics["ag_news"]["baseline"]),
                "attention": summarize(metrics["ag_news"]["attention"]),
            },
            "online_news": {
                "baseline": summarize(metrics["online_news"]["baseline"]),
                "attention": summarize(metrics["online_news"]["attention"]),
            },
        },
        "statistical_tests": stats_results,
        "attention_entropy_analysis": entropy_analysis,
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(MODEL_OUTPUT_DIR / "attention_entropy_samples.json", "w") as f:
        json.dump(
            {
                "ag_news": {
                    "entropy": ag_entropy_agg["entropy"][:2000],
                    "correct": ag_entropy_agg["correct"][:2000],
                    "confidence": ag_entropy_agg["confidence"][:2000],
                },
                "online_news": {
                    "entropy": on_entropy_agg["entropy"][:2000],
                    "correct": on_entropy_agg["correct"][:2000],
                    "score": on_entropy_agg["confidence"][:2000],
                },
            },
            f,
            indent=2,
        )

    make_plots(metrics, ag_entropy_agg, on_entropy_agg)

    logging.info("Finished. Runtime: %.2f minutes", runtime_min)


if __name__ == "__main__":
    main()
