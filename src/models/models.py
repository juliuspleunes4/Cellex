"""
CELLEX CANCER DETECTION SYSTEM - MODEL ARCHITECTURES
===================================================
Professional deep learning models for cancer detection in X-ray images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import timm
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from config.config import get_config


class AttentionBlock(nn.Module):
    """Attention mechanism for focusing on relevant image regions."""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class CellexEfficientNet(nn.Module):
    """
    Professional EfficientNet-based model for cancer detection.
    
    Features:
    - Pre-trained EfficientNet backbone
    - Custom classification head
    - Attention mechanism
    - Dropout for regularization
    - Batch normalization
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet-b0",
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        
        super(CellexEfficientNet, self).__init__()
        
        self.logger = get_logger("CellexEfficientNet")
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove final classifier
            global_pool=""  # Remove global pooling
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            self.feature_size = features.shape[-1]
        
        self.logger.info(f"🧠 Backbone: {model_name}")
        self.logger.info(f"📐 Feature dimensions: {self.feature_dim}")
        self.logger.info(f"📐 Feature map size: {self.feature_size}x{self.feature_size}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.logger.info("🔒 Backbone frozen")
        
        # Attention mechanism
        self.attention = AttentionBlock(self.feature_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.model_info(
            f"{model_name} (Cellex Enhanced)",
            self.trainable_params
        )
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global pooling
        pooled_features = self.global_pool(attended_features)
        pooled_features = pooled_features.flatten(1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
    
    def get_attention_map(self, x):
        """Get attention visualization for interpretability."""
        features = self.backbone(x)
        
        # Get attention weights
        attention_weights = self.attention.conv1(features)
        attention_weights = F.relu(attention_weights)
        attention_weights = self.attention.conv2(attention_weights)
        attention_weights = self.attention.sigmoid(attention_weights)
        
        return attention_weights


class CellexResNet(nn.Module):
    """ResNet-based model for cancer detection."""
    
    def __init__(self,
                 model_name: str = "resnet50",
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 pretrained: bool = True):
        
        super(CellexResNet, self).__init__()
        
        self.logger = get_logger("CellexResNet")
        
        # Load pre-trained ResNet
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.model_info(f"{model_name} (Cellex)", self.trainable_params)
    
    def forward(self, x):
        return self.backbone(x)


class CellexEnsemble(nn.Module):
    """
    Ensemble model combining multiple architectures for robust predictions.
    """
    
    def __init__(self, 
                 model_configs: List[Dict],
                 num_classes: int = 2,
                 ensemble_method: str = "average"):
        
        super(CellexEnsemble, self).__init__()
        
        self.logger = get_logger("CellexEnsemble")
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        
        # Create individual models
        self.models = nn.ModuleList()
        
        for i, config in enumerate(model_configs):
            model_type = config.get("type", "efficientnet")
            
            if model_type == "efficientnet":
                model = CellexEfficientNet(
                    model_name=config.get("name", "efficientnet-b0"),
                    num_classes=num_classes,
                    dropout_rate=config.get("dropout", 0.2),
                    pretrained=config.get("pretrained", True),
                    freeze_backbone=config.get("freeze_backbone", False)
                )
            elif model_type == "resnet":
                model = CellexResNet(
                    model_name=config.get("name", "resnet50"),
                    num_classes=num_classes,
                    dropout_rate=config.get("dropout", 0.2),
                    pretrained=config.get("pretrained", True)
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.models.append(model)
            self.logger.info(f"🔗 Added model {i+1}: {config.get('name', 'Unknown')}")
        
        # Ensemble combination layer (if using learned ensemble)
        if ensemble_method == "learned":
            self.combination_layer = nn.Sequential(
                nn.Linear(num_classes * len(model_configs), 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        
        total_params = sum(sum(p.numel() for p in model.parameters()) for model in self.models)
        self.logger.model_info(f"Ensemble ({len(model_configs)} models)", total_params)
    
    def forward(self, x):
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        if self.ensemble_method == "average":
            # Simple averaging
            ensemble_output = torch.stack(outputs).mean(dim=0)
        
        elif self.ensemble_method == "weighted":
            # Weighted averaging (weights learned during training)
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_outputs = [w * output for w, output in zip(weights, outputs)]
            ensemble_output = torch.stack(weighted_outputs).sum(dim=0)
        
        elif self.ensemble_method == "learned":
            # Learned combination
            concatenated = torch.cat(outputs, dim=1)
            ensemble_output = self.combination_layer(concatenated)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical imaging.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_model(config) -> nn.Module:
    """Factory function to create models based on configuration."""
    logger = get_logger("ModelFactory")
    
    if config.model.use_ensemble:
        # Create ensemble model
        model_configs = [
            {"type": "efficientnet", "name": "efficientnet-b0", "dropout": 0.2},
            {"type": "resnet", "name": "resnet50", "dropout": 0.2},
            {"type": "efficientnet", "name": "efficientnet-b1", "dropout": 0.25},
        ]
        
        model = CellexEnsemble(
            model_configs=model_configs,
            num_classes=config.model.num_classes,
            ensemble_method="average"
        )
        
    else:
        # Create single model
        backbone = config.model.backbone.lower()
        
        if "efficientnet" in backbone:
            model = CellexEfficientNet(
                model_name=config.model.backbone,
                num_classes=config.model.num_classes,
                dropout_rate=config.model.dropout_rate,
                pretrained=config.model.pretrained,
                freeze_backbone=config.model.freeze_backbone
            )
        
        elif "resnet" in backbone:
            model = CellexResNet(
                model_name=config.model.backbone,
                num_classes=config.model.num_classes,
                dropout_rate=config.model.dropout_rate,
                pretrained=config.model.pretrained
            )
        
        else:
            raise ValueError(f"Unsupported backbone: {config.model.backbone}")
    
    logger.success(f"✅ Model created: {type(model).__name__}")
    return model


def create_loss_function(config) -> nn.Module:
    """Create loss function based on configuration."""
    if config.training.loss_function == "focal_loss":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif config.training.loss_function == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif config.training.loss_function == "weighted_cross_entropy":
        if config.training.class_weights:
            weights = torch.tensor(config.training.class_weights)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.training.loss_function}")


if __name__ == "__main__":
    # Test model creation
    from config.config import CellexConfig
    
    config = CellexConfig()
    logger = get_logger("ModelTest")
    
    logger.welcome()
    logger.section("MODEL ARCHITECTURE TESTING")
    
    # Test single model
    logger.subsection("Testing EfficientNet Model")
    model = create_model(config)
    
    # Test input
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    logger.success(f"✅ Model output shape: {output.shape}")
    
    # Test loss function
    logger.subsection("Testing Loss Function")
    loss_fn = create_loss_function(config)
    dummy_targets = torch.randint(0, 2, (2,))
    loss = loss_fn(output, dummy_targets)
    logger.success(f"✅ Loss value: {loss.item():.4f}")
    
    logger.success("🎉 All model tests passed!")