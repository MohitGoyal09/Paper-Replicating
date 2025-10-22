import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import math
import time
from pathlib import Path
from tqdm import tqdm
import json

from model import Transformer, ModelArgs

class TextDataset(Dataset):
    
    def __init__(self, text_data, block_size, vocab_size):
        self.data = text_data
        self.block_size = block_size
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def get_batch(data, batch_size, block_size, device):
    """Generate a batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """Estimate loss on train and validation sets"""
    model.eval()
    losses = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses[split] = 0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with torch.no_grad():
                logits = model(x, 0)  # start_pos = 0 for training
                logits = logits.view(-1, logits.size(-1))
                y = y.view(-1)
                loss = F.cross_entropy(logits, y)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    
    model.train()
    return losses

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filepath}")
        return epoch, loss
    return 0, float('inf')

def train_model(
    model,
    train_data,
    val_data,
    device,
    batch_size=8,
    block_size=128,
    learning_rate=3e-4,
    max_iters=5000,
    eval_interval=100,
    save_interval=500,
    checkpoint_path="checkpoint.pt"
):
    """Train the LLaMA model"""
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Load checkpoint if exists
    start_iter, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Training loop
    model.train()
    pbar = tqdm(range(start_iter, max_iters), desc="Training")
    
    for iter_num in pbar:
        # Get batch
        x, y = get_batch(train_data, batch_size, block_size, device)
        
        # Forward pass
        logits = model(x, 0)  # start_pos = 0 for training
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluation
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_data, val_data, 100, batch_size, block_size, device)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save checkpoint if validation loss improved
            if losses['val'] < best_loss:
                best_loss = losses['val']
                save_checkpoint(model, optimizer, iter_num, losses['val'], checkpoint_path)
        
        # Save checkpoint at intervals
        if iter_num % save_interval == 0 and iter_num > 0:
            save_checkpoint(model, optimizer, iter_num, loss.item(), f"checkpoint_step_{iter_num}.pt")

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    model_args = ModelArgs(
        dim=512,  # Smaller for training
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=10000,  # Will be set based on data
        max_seq_len=128,
        device=device
    )
    
    # Load and prepare data
    print("Loading data...")
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Update vocab size
    model_args.vocab_size = vocab_size
    
    # Encode text
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print("Creating model...")
    model = Transformer(model_args).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        batch_size=8,
        block_size=128,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=100,
        save_interval=500
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
