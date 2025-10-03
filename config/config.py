"""
CELLEX CANCER DETECTION SYSTEM - CONFIGURATION
==============================================
Professional configuration management for the Cellex AI system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    
    # Kaggle datasets for cancer detection
    datasets: List[str] = None
    
    # Data directories
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data augmentation
    augmentation_enabled: bool = True
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "mohamedhanyyy/chest-ctscan-images",  # Chest CT-Scan Cancer Detection
                "andrewmvd/lung-and-colon-cancer-histopathological-images",  # Lung Cancer Histopathological
                "sartajbhuvaji/brain-tumor-classification-mri",  # Brain Tumor Classification
                "kmader/skin-cancer-mnist-ham10000",  # Skin Cancer Classification
            ]


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    backbone: str = "efficientnet_b0"
    num_classes: int = 2  # Normal, Cancer
    dropout_rate: float = 0.2
    
    # Pre-trained weights
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Model ensemble
    use_ensemble: bool = False
    ensemble_models: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = [
                "efficientnet_b0",
                "resnet50",
                "densenet121"
            ]


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    batch_size: int = 64  # Increased for better GPU utilization
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    
    # Learning rate scheduler
    scheduler: str = "cosine"
    scheduler_params: Dict = None
    
    # Loss function
    loss_function: str = "focal_loss"  # Better for medical imaging
    class_weights: Optional[List[float]] = None
    
    # Regularization
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {
                "T_max": self.num_epochs,
                "eta_min": 1e-6
            }


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    # Model loading
    model_path: str = "models/best_model.pth"
    device: str = "auto"  # auto, cpu, cuda
    
    # Inference parameters
    batch_size: int = 16
    threshold: float = 0.5
    
    # Output format
    return_probabilities: bool = True
    return_attention_maps: bool = True
    
    # Post-processing
    apply_tta: bool = True  # Test Time Augmentation
    tta_steps: int = 8


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    
    # Experiment tracking
    use_mlflow: bool = False
    use_wandb: bool = False
    experiment_name: str = "cellex_cancer_detection"
    
    # Logging levels
    log_level: str = "INFO"
    log_file: str = "logs/cellex.log"
    
    # Metrics tracking
    track_gradients: bool = True
    log_images: bool = True
    log_frequency: int = 10  # More frequent updates for better dev experience


@dataclass
class CellexConfig:
    """Main Cellex configuration."""
    
    # Sub-configurations
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    logging: LoggingConfig = None
    
    # Project settings
    project_name: str = "Cellex Cancer Detection"
    version: str = "1.0.0"
    seed: int = 42
    
    # Directories
    project_root: str = None
    
    def __post_init__(self):
        # Initialize sub-configs
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
            
        # Set project root
        if self.project_root is None:
            self.project_root = str(Path(__file__).parent.parent)
    
    def save_config(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        
        def convert_tuples_to_lists(obj):
            """Convert tuples to lists recursively for YAML serialization."""
            if isinstance(obj, dict):
                return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_tuples_to_lists(item) for item in obj]
            else:
                return obj
        
        config_dict = {
            'project_name': self.project_name,
            'version': self.version,
            'seed': self.seed,
            'data': convert_tuples_to_lists(self.data.__dict__),
            'model': convert_tuples_to_lists(self.model.__dict__),
            'training': convert_tuples_to_lists(self.training.__dict__),
            'inference': convert_tuples_to_lists(self.inference.__dict__),
            'logging': convert_tuples_to_lists(self.logging.__dict__)
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        config.project_name = config_dict.get('project_name', config.project_name)
        config.version = config_dict.get('version', config.version)
        config.seed = config_dict.get('seed', config.seed)
        
        # Update sub-configs
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                # Convert image_size list back to tuple
                if key == 'image_size' and isinstance(value, list):
                    value = tuple(value)
                setattr(config.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config.training, key, value)
        
        if 'inference' in config_dict:
            for key, value in config_dict['inference'].items():
                setattr(config.inference, key, value)
        
        if 'logging' in config_dict:
            for key, value in config_dict['logging'].items():
                setattr(config.logging, key, value)
        
        return config


# Default configuration instance
default_config = CellexConfig()


def get_config() -> CellexConfig:
    """Get the current configuration."""
    config_path = os.environ.get('CELLEX_CONFIG_PATH', 'config/config.yaml')
    
    if os.path.exists(config_path):
        return CellexConfig.load_config(config_path)
    else:
        return default_config


if __name__ == "__main__":
    # Create default configuration file
    config = CellexConfig()
    os.makedirs("config", exist_ok=True)
    config.save_config("config/config.yaml")
    print("[SUCCESS] Default configuration saved to config/config.yaml")