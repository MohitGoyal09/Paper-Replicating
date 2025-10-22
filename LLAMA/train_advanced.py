

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import math
import time
import json
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional

from model import Transformer, ModelArgs
from data_utils import prepare_data, create_dataloaders
from config import TrainingConfig, get_config

class AdvancedTrainer:
    """Advanced trainer with additional features"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Setup mixed precision
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def setup_logging(self):
        """Setup logging directory and wandb if available"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Try to initialize wandb
        try:
            wandb.init(
                project="llama-training",
                config=vars(self.config),
                name=f"llama-{self.config.dim}d-{int(time.time())}"
            )
            self.use_wandb = True
        except:
            self.use_wandb = False
            print("Wandb not available, using local logging only")
    
    def get_lr(self, iter_num: int) -> float:
        """Get learning rate with warmup and decay"""
        if iter_num < self.config.warmup_iters:
            return self.config.learning_rate * iter_num / self.config.warmup_iters
        if iter_num > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (iter_num - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def estimate_loss(self, model, train_loader, val_loader, eval_iters: int) -> Dict[str, float]:
        """Estimate loss on train and validation sets"""
        model.eval()
        losses = {}
        
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses[split] = 0
            for i, (x, y) in enumerate(loader):
                if i >= eval_iters:
                    break
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            logits = model(x, 0)
                            logits = logits.view(-1, logits.size(-1))
                            y = y.view(-1)
                            loss = nn.functional.cross_entropy(logits, y)
                    else:
                        logits = model(x, 0)
                        logits = logits.view(-1, logits.size(-1))
                        y = y.view(-1)
                        loss = nn.functional.cross_entropy(logits, y)
                losses[split] += loss.item()
            losses[split] /= min(eval_iters, len(loader))
        
        model.train()
        return losses
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'loss': loss,
            'config': vars(self.config),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = f"{self.config.checkpoint_path}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{self.config.checkpoint_path}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
        
        # Save last checkpoint
        if self.config.save_last:
            last_path = f"{self.config.checkpoint_path}_last.pt"
            torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, model, optimizer, scheduler=None):
        """Load model checkpoint"""
        if os.path.exists(f"{self.config.checkpoint_path}.pt"):
            checkpoint = torch.load(f"{self.config.checkpoint_path}.pt", map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.training_history = checkpoint.get('training_history', [])
            self.best_val_loss = checkpoint.get('loss', float('inf'))
            
            print(f"Checkpoint loaded from {self.config.checkpoint_path}.pt")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
        return 0, float('inf')
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # Get current learning rate
            current_lr = self.get_lr(epoch * len(train_loader) + batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x, 0)
                    logits = logits.view(-1, logits.size(-1))
                    y = y.view(-1)
                    loss = nn.functional.cross_entropy(logits, y)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = model(x, 0)
                logits = logits.view(-1, logits.size(-1))
                y = y.view(-1)
                loss = nn.functional.cross_entropy(logits, y)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log metrics
            if batch_idx % self.config.log_interval == 0:
                log_data = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'lr': current_lr
                }
                self.training_history.append(log_data)
                
                if self.use_wandb:
                    wandb.log(log_data)
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print(f"Using device: {self.device}")
        
        # Load data
        print("Loading data...")
        train_data, val_data, stoi, itos, vocab_size = prepare_data(
            self.config.data_path, 
            self.config.vocab_path
        )
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, 
            self.config.block_size, 
            self.config.batch_size
        )
        
        # Update vocab size in config
        self.config.vocab_size = vocab_size
        
        # Create model
        print("Creating model...")
        model_args = ModelArgs(
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            device=str(self.device)
        )
        
        self.model = Transformer(model_args).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.lr_decay_iters,
            eta_min=self.config.min_lr
        )
        
        # Load checkpoint if resuming
        start_epoch = 0
        if self.config.resume_from_checkpoint:
            start_epoch, self.best_val_loss = self.load_checkpoint(
                self.model, self.optimizer, self.scheduler
            )
        
        # Training loop
        print("Starting training...")
        for epoch in range(start_epoch, self.config.max_iters):
            # Train epoch
            train_loss = self.train_epoch(
                self.model, train_loader, self.optimizer, self.scheduler, epoch
            )
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_losses = self.estimate_loss(
                    self.model, train_loader, val_loader, self.config.eval_iters
                )
                
                print(f"Epoch {epoch}: train_loss={val_losses['train']:.4f}, val_loss={val_losses['val']:.4f}")
                
                # Save checkpoint if validation loss improved
                is_best = val_losses['val'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['val']
                
                self.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, val_losses['val'], is_best
                )
                
                # Log validation metrics
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': val_losses['train'],
                        'val_loss': val_losses['val'],
                        'best_val_loss': self.best_val_loss
                    })
            
            # Save checkpoint at intervals
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, train_loss, False
                )
        
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(
            self.model, self.optimizer, self.scheduler, 
            epoch, train_loss, False
        )

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced LLaMA Training")
    parser.add_argument("--config", type=str, default="medium", 
                        help="Model size: small, medium, large")
    parser.add_argument("--data_path", type=str, default="data/input.txt",
                        help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Maximum iterations")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_iters:
        config.max_iters = args.max_iters
    if args.use_amp:
        config.use_amp = True
    
    # Create trainer and start training
    trainer = AdvancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
