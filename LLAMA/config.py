from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    
    # Model architecture
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 10000
    max_seq_len: int = 128
    
    # Training hyperparameters
    batch_size: int = 8
    block_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    
    # Evaluation and saving
    eval_interval: int = 100
    save_interval: int = 500
    eval_iters: int = 100
    
    # Data
    data_path: str = "data/input.txt"
    vocab_path: str = "data/vocab.json"
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"
    resume_from_checkpoint: bool = True
    
    # Advanced training options
    gradient_clip_val: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 3e-5
    
    # Mixed precision training
    use_amp: bool = False
    
    # Logging
    log_interval: int = 10
    log_dir: str = "logs"
    
    # Model saving
    save_best_only: bool = True
    save_last: bool = True

@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    # Model loading
    checkpoint_path: str = "checkpoint.pt"
    model_path: str = "model.pt"
    
    # Generation parameters
    temperature: float = 0.6
    top_p: float = 0.9
    max_gen_len: int = 128
    num_samples: int = 1
    
    # Device
    device: str = "auto"
    
    # Tokenizer
    tokenizer_path: str = "tokenizer.model"
    
    # Output
    output_file: str = "generated_text.txt"

# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()

# Model size presets
SMALL_MODEL = TrainingConfig(
    dim=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=64,
    batch_size=16,
    block_size=64
)

MEDIUM_MODEL = TrainingConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=128,
    batch_size=8,
    block_size=128
)

LARGE_MODEL = TrainingConfig(
    dim=1024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=8,
    max_seq_len=256,
    batch_size=4,
    block_size=256
)

def get_config(model_size: str = "medium") -> TrainingConfig:
   
    configs = {
        "small": SMALL_MODEL,
        "medium": MEDIUM_MODEL,
        "large": LARGE_MODEL
    }
    return configs.get(model_size, MEDIUM_MODEL)

def update_config(config: TrainingConfig, **kwargs) -> TrainingConfig:
 
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter '{key}'")
    return config
