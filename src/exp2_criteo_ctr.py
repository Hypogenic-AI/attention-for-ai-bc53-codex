import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import os

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load and Preprocess Data
def load_and_preprocess_criteo():
    print("Loading Criteo data...")
    df = pd.read_csv("datasets/criteo/criteo_20k.csv")
    
    dense_features = [f"I{i}" for i in range(1, 14)]
    sparse_features = [f"C{i}" for i in range(1, 27)]
    
    # Fill missing values
    df[dense_features] = df[dense_features].fillna(0)
    df[sparse_features] = df[sparse_features].fillna("-1")
    
    # Preprocess dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    
    # Preprocess sparse features
    sparse_feature_vocabs = []
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat].astype(str))
        sparse_feature_vocabs.append(len(lbe.classes_))
        
    return df, dense_features, sparse_features, sparse_feature_vocabs

# Dataset for CTR
class CTRDataset(Dataset):
    def __init__(self, df, dense_features, sparse_features):
        self.dense = df[dense_features].values.astype(np.float32)
        self.sparse = df[sparse_features].values.astype(np.int64)
        self.label = df['label'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return torch.tensor(self.dense[idx]), torch.tensor(self.sparse[idx]), torch.tensor(self.label[idx])

# Models
class MLPModel(nn.Module):
    def __init__(self, dense_dim, sparse_vocabs, embed_dim=8):
        super(MLPModel, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, embed_dim) for vocab in sparse_vocabs
        ])
        total_dim = dense_dim + len(sparse_vocabs) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, dense, sparse):
        sparse_embs = [emb(sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_embs = torch.cat(sparse_embs, dim=1)
        x = torch.cat([dense, sparse_embs], dim=1)
        return self.mlp(x).squeeze(1)

class AttentionMLPModel(nn.Module):
    def __init__(self, dense_dim, sparse_vocabs, embed_dim=8):
        super(AttentionMLPModel, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, embed_dim) for vocab in sparse_vocabs
        ])
        self.dense_proj = nn.Linear(dense_dim, dense_dim * embed_dim) # Not used, keep it simple
        
        # Simple self-attention over embeddings
        self.num_sparse = len(sparse_vocabs)
        self.query = nn.Linear(embed_dim, 16)
        self.key = nn.Linear(embed_dim, 16)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        total_dim = dense_dim + self.num_sparse * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, dense, sparse):
        sparse_embs = [emb(sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        sparse_embs = torch.stack(sparse_embs, dim=1) # (batch, num_sparse, embed_dim)
        
        # Simple Attention Mechanism
        # Instead of full self-attention, we weight each feature
        q = self.query(sparse_embs)
        k = self.key(sparse_embs)
        v = self.value(sparse_embs)
        
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(16)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attended_sparse = torch.matmul(attn_weights, v) # (batch, num_sparse, embed_dim)
        attended_sparse = attended_sparse.view(attended_sparse.size(0), -1)
        
        x = torch.cat([dense, attended_sparse], dim=1)
        return self.mlp(x).squeeze(1), attn_weights

# Training and Evaluation
def train_and_eval(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for dense, sparse, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            dense, sparse, y = dense.to(device), sparse.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(dense, sparse)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        total_loss_val = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for dense, sparse, y in val_loader:
                dense, sparse, y = dense.to(device), sparse.to(device), y.to(device)
                outputs = model(dense, sparse)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                total_loss_val += criterion(logits, y).item()
                preds = (logits > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}, Val Acc: {correct/total:.4f}")
    
    return correct/total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df, dense_features, sparse_features, sparse_vocabs = load_and_preprocess_criteo()
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = CTRDataset(train_df, dense_features, sparse_features)
    val_dataset = CTRDataset(val_df, dense_features, sparse_features)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print("\n--- Training MLP (Non-Attention) ---")
    mlp_model = MLPModel(len(dense_features), sparse_vocabs)
    mlp_acc = train_and_eval(mlp_model, train_loader, val_loader, epochs=5, device=device)
    
    print("\n--- Training AttentionMLP (Attention) ---")
    attn_model = AttentionMLPModel(len(dense_features), sparse_vocabs)
    attn_acc = train_and_eval(attn_model, train_loader, val_loader, epochs=5, device=device)
    
    print(f"\nFinal Results:")
    print(f"MLP Acc: {mlp_acc:.4f}")
    print(f"AttentionMLP Acc: {attn_acc:.4f}")
    
    # Save models
    torch.save(mlp_model.state_dict(), "results/mlp_criteo.pth")
    torch.save(attn_model.state_dict(), "results/attn_criteo.pth")
    
    # Save results
    results = {
        "mlp_acc": mlp_acc,
        "attn_acc": attn_acc
    }
    os.makedirs("results", exist_ok=True)
    pd.Series(results).to_json("results/criteo_results.json")

if __name__ == "__main__":
    main()
