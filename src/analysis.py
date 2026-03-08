import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from exp1_movielens_seq import TransformerModel, load_and_preprocess_ml, SeqDataset
from exp2_criteo_ctr import AttentionMLPModel, load_and_preprocess_criteo, CTRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_entropy(last_attn):
    # last_attn shape: (batch, seq_len)
    # Add small epsilon for log
    entropy = -torch.sum(last_attn * torch.log(last_attn + 1e-9), dim=-1)
    return entropy

def analyze_movielens_attention():
    print("\n--- Analyzing MovieLens Attention ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_seqs, num_items = load_and_preprocess_ml()
    
    # Take a subset for evaluation
    val_seqs = user_seqs[-1000:]
    val_dataset = SeqDataset(val_seqs)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = TransformerModel(num_items, embed_dim=64, nhead=4, nhid=128, nlayers=2)
    model.load_state_dict(torch.load("results/trans_movielens.pth"))
    model.to(device)
    model.eval()
    
    entropies = []
    correctness = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            logits, all_attn = model(x)
            
            # Use last layer attention
            last_layer_attn = all_attn[-1] # (batch, seq_len, seq_len)
            # Attention of last item to all previous items
            last_attn = last_layer_attn[:, -1, :] 
            
            ent = compute_entropy(last_attn)
            entropies.extend(ent.cpu().numpy())
            
            _, preds = torch.topk(logits, 1, dim=1)
            corr = (preds.squeeze(1) == y).float()
            correctness.extend(corr.cpu().numpy())
    
    df_results = pd.DataFrame({"entropy": entropies, "correct": correctness})
    # Group by correctness and compare entropy
    mean_entropy = df_results.groupby("correct")["entropy"].mean()
    print(f"Mean Attention Entropy by Correctness:\n{mean_entropy}")
    
    plt.figure(figsize=(8, 5))
    df_results.boxplot(column="entropy", by="correct")
    plt.title("Attention Entropy vs Prediction Correctness (MovieLens)")
    plt.suptitle("")
    plt.savefig("results/plots/movielens_entropy.png")
    plt.close()

def analyze_criteo_attention():
    print("\n--- Analyzing Criteo Attention ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df, dense_features, sparse_features, sparse_vocabs = load_and_preprocess_criteo()
    
    val_df = df.iloc[-2000:]
    val_dataset = CTRDataset(val_df, dense_features, sparse_features)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = AttentionMLPModel(len(dense_features), sparse_vocabs)
    model.load_state_dict(torch.load("results/attn_criteo.pth"))
    model.to(device)
    model.eval()
    
    all_attn_weights = []
    with torch.no_grad():
        for dense, sparse, y in tqdm(val_loader):
            dense, sparse, y = dense.to(device), sparse.to(device), y.to(device)
            _, attn_weights = model(dense, sparse)
            # attn_weights: (batch, num_sparse, num_sparse)
            # Take the diagonal or mean weights per feature
            # Let's take the mean attention received by each feature
            feature_attn = attn_weights.mean(dim=1) # (batch, num_sparse)
            all_attn_weights.append(feature_attn.cpu().numpy())
            
    mean_feature_attn = np.concatenate(all_attn_weights, axis=0).mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sparse_features)), mean_feature_attn)
    plt.xticks(range(len(sparse_features)), sparse_features, rotation=90)
    plt.title("Average Attention Weight per Feature (Criteo)")
    plt.xlabel("Sparse Feature")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig("results/plots/criteo_feature_attention.png")
    plt.close()
    print("Criteo attention visualization saved.")

if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)
    analyze_movielens_attention()
    analyze_criteo_attention()
