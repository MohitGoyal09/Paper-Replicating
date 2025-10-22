import torch
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

def download_tinyshakespeare():
  
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    input_file = data_dir / "input.txt"
    
    if not input_file.exists():
        print("Downloading TinyShakespeare dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, input_file)
        print(f"Dataset downloaded to {input_file}")
    else:
        print(f"Dataset already exists at {input_file}")
    
    return input_file

def create_vocabulary(text: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Create vocabulary from text"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return stoi, itos, vocab_size

def encode_text(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    """Encode text to tensor"""
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def decode_tensor(tensor: torch.Tensor, itos: Dict[int, str]) -> str:
    """Decode tensor to text"""
    return ''.join([itos[i.item()] for i in tensor])

def save_vocabulary(stoi: Dict[str, int], itos: Dict[int, str], filepath: str):
    "
    vocab_data = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(stoi)
    }
    with open(filepath, 'w') as f:
        json.dump(vocab_data, f, indent=2)

def load_vocabulary(filepath: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    
    with open(filepath, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

def prepare_data(data_path: str = "data/input.txt", vocab_path: str = "data/vocab.json"):
    
    # Download data if needed
    if not os.path.exists(data_path):
        download_tinyshakespeare()
    
    # Load text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text)} characters")
    
    # Create vocabulary
    stoi, itos, vocab_size = create_vocabulary(text)
    
    # Save vocabulary
    save_vocabulary(stoi, itos, vocab_path)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary saved to {vocab_path}")
    
    # Encode text
    data = encode_text(text, stoi)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    
    return train_data, val_data, stoi, itos, vocab_size

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class TextDataset(torch.utils.data.Dataset):
   
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

def create_dataloaders(train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int):

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
   
    train_data, val_data, stoi, itos, vocab_size = prepare_data()
    print("Data preparation completed successfully!")
