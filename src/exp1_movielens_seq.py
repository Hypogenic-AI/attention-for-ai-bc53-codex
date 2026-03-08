import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load and Preprocess Data
def load_and_preprocess_ml():
    print("Loading MovieLens data...")
    df = pd.read_csv("datasets/movielens/ratings_full.csv")
    
    # Simple preprocessing
    # Assign row index as "pseudo-timestamp" to preserve order
    df['timestamp'] = df.index
    
    # Map users and movies to IDs
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    movie_map = {id: i+1 for i, id in enumerate(df['movie_id'].unique())} # i+1 for padding 0
    
    df['user'] = df['user_id'].map(user_map)
    df['movie'] = df['movie_id'].map(movie_map)
    
    num_users = len(user_map)
    num_movies = len(movie_map) + 1 # +1 for padding
    
    # Group by user and create sequences
    print("Creating sequences...")
    user_seqs = df.sort_values(['user', 'timestamp']).groupby('user')['movie'].apply(list).tolist()
    
    # Filter sequences to have at least 5 items
    user_seqs = [seq for seq in user_seqs if len(seq) >= 5]
    
    return user_seqs, num_movies

# Dataset for Sequential Recommendation
class SeqDataset(Dataset):
    def __init__(self, sequences, max_len=20):
        self.sequences = sequences
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Target is the last item, Input is the previous ones
        target = seq[-1]
        input_seq = seq[:-1]
        
        # Truncate or pad input_seq
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
        else:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
            
        return torch.tensor(input_seq), torch.tensor(target)

# Models
class GRUModel(nn.Module):
    def __init__(self, num_items, embed_dim, hidden_dim):
        super(GRUModel, self).__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)
        
    def forward(self, x):
        emb = self.item_emb(x)
        out, _ = self.gru(emb)
        last_out = out[:, -1, :] # Take the last hidden state
        logits = self.fc(last_out)
        return logits

class TransformerModel(nn.Module):
    def __init__(self, num_items, embed_dim, nhead, nhid, nlayers, max_len=20):
        super(TransformerModel, self).__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        
        # Use custom layer to get attention weights
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, nhead, batch_first=True) for _ in range(nlayers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(nlayers)])
        
        self.fc = nn.Linear(embed_dim, num_items)
        self.max_len = max_len
        
    def forward(self, x):
        positions = torch.arange(0, self.max_len, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        emb = self.item_emb(x) + self.pos_emb(positions)
        
        padding_mask = (x == 0)
        
        all_attn = []
        out = emb
        for attn_layer, norm in zip(self.layers, self.norms):
            attn_out, attn_weights = attn_layer(out, out, out, key_padding_mask=padding_mask)
            out = norm(out + attn_out)
            all_attn.append(attn_weights)
            
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits, all_attn

# Training and Evaluation
def train_and_eval(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
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
        hits = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                # Hit@10
                _, indices = torch.topk(logits, 10, dim=1)
                hits += (indices == y.view(-1, 1)).any(dim=1).sum().item()
                total += y.size(0)
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}, Hit@10: {hits/total:.4f}")
    
    return hits/total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    user_seqs, num_items = load_and_preprocess_ml()
    
    # Split users
    train_seqs, val_seqs = train_test_split(user_seqs, test_size=0.2, random_state=42)
    
    train_dataset = SeqDataset(train_seqs)
    val_dataset = SeqDataset(val_seqs)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print("\n--- Training GRU (Non-Attention) ---")
    gru_model = GRUModel(num_items, embed_dim=64, hidden_dim=64)
    gru_hit = train_and_eval(gru_model, train_loader, val_loader, epochs=3, device=device)
    
    print("\n--- Training Transformer (Attention) ---")
    trans_model = TransformerModel(num_items, embed_dim=64, nhead=4, nhid=128, nlayers=2)
    trans_hit = train_and_eval(trans_model, train_loader, val_loader, epochs=3, device=device)
    
    print(f"\nFinal Results:")
    print(f"GRU Hit@10: {gru_hit:.4f}")
    print(f"Transformer Hit@10: {trans_hit:.4f}")
    
    # Save models
    torch.save(gru_model.state_dict(), "results/gru_movielens.pth")
    torch.save(trans_model.state_dict(), "results/trans_movielens.pth")
    
    # Save results
    results = {
        "gru_hit": gru_hit,
        "trans_hit": trans_hit
    }
    os.makedirs("results", exist_ok=True)
    pd.Series(results).to_json("results/movielens_results.json")

if __name__ == "__main__":
    main()
