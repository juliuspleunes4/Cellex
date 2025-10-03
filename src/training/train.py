"""
CELLEX CANCER DETECTION SYSTEM - TRAINING PIPELINE
=================================================
Professional training system for cancer detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import json
import os
import signal
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.models.models import create_model, create_loss_function
from src.data.data_loader import create_data_loaders
from config.config import get_config

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CellexMetrics:
    """Professional metrics tracking for cancer detection."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.correct = 0
        self.total = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        # Convert to predictions
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Update counts
        batch_correct = (pred_classes == targets).sum().item()
        self.correct += batch_correct
        self.total += targets.size(0)
        self.losses.append(loss)
        
        # Update confusion matrix components
        for pred, target in zip(pred_classes, targets):
            if pred == 1 and target == 1:
                self.true_positives += 1
            elif pred == 1 and target == 0:
                self.false_positives += 1
            elif pred == 0 and target == 0:
                self.true_negatives += 1
            elif pred == 0 and target == 1:
                self.false_negatives += 1
    
    def get_accuracy(self) -> float:
        """Calculate accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def get_precision(self) -> float:
        """Calculate precision."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def get_recall(self) -> float:
        """Calculate recall (sensitivity)."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def get_specificity(self) -> float:
        """Calculate specificity."""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 0.0
    
    def get_f1_score(self) -> float:
        """Calculate F1 score."""
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def get_average_loss(self) -> float:
        """Calculate average loss."""
        return np.mean(self.losses) if self.losses else 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class CellexTrainer:
    """Professional training system for Cellex cancer detection."""
    
    def __init__(self, config=None, resume_from=None):
        self.config = config or get_config()
        self.logger = get_logger("CellexTrainer")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🖥️  Device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Create directories
        self.model_dir = Path("models")
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("checkpoints")
        
        for dir_path in [self.model_dir, self.log_dir, self.checkpoint_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Store resume path for later use
        self.resume_from = resume_from
        
        # Setup graceful shutdown
        self.interrupted = False
        self.interrupt_save_vars = None
        
        # Initialize experiment tracking
        self._init_experiment_tracking()
    
    def _init_experiment_tracking(self):
        """Initialize MLflow and Weights & Biases tracking."""
        # MLflow setup
        if MLFLOW_AVAILABLE and self.config.logging.use_mlflow:
            try:
                mlflow.set_experiment(self.config.logging.experiment_name)
                self.logger.success("✅ MLflow initialized")
            except Exception as e:
                self.logger.warning(f"⚠️  MLflow initialization failed: {str(e)}")
        
        # Weights & Biases setup
        if WANDB_AVAILABLE and self.config.logging.use_wandb:
            try:
                wandb.init(
                    project=self.config.logging.experiment_name,
                    config=self.config.__dict__
                )
                self.logger.success("✅ Weights & Biases initialized")
            except Exception as e:
                self.logger.warning(f"⚠️  Weights & Biases initialization failed: {str(e)}")
    
    def train(self, data_dir: str) -> Dict:
        """Main training function."""
        self.logger.welcome()
        self.logger.section("CELLEX CANCER DETECTION TRAINING")
        
        start_time = time.time()
        
        # Load data
        self.logger.subsection("Loading Data")
        train_loader, val_loader, test_loader = create_data_loaders(data_dir, self.config)
        
        # Create model
        self.logger.subsection("Creating Model")
        model = create_model(self.config)
        model = model.to(self.device)
        
        # Create loss function and optimizer
        criterion = create_loss_function(self.config)
        criterion = criterion.to(self.device)
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        
        # Handle checkpoint resume
        start_epoch = 1
        if self.resume_from:
            try:
                resume_epoch = self.load_checkpoint(self.resume_from, model, optimizer, scheduler)
                start_epoch = resume_epoch + 1
                self.logger.info(f"🔄 Resuming training from epoch {start_epoch}")
            except Exception as e:
                self.logger.error(f"❌ Failed to resume from checkpoint: {str(e)}")
                self.logger.info("🔄 Starting fresh training instead...")
                start_epoch = 1
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=0.001
        )
        
        # Setup graceful shutdown for interruption
        def signal_handler(signum, frame):
            self.logger.info("\n⚠️ Training interruption requested (Ctrl+C)")
            self.interrupted = True
            if self.interrupt_save_vars:
                model_ref, optimizer_ref, scheduler_ref, current_epoch = self.interrupt_save_vars
                self.logger.info(f"💾 Saving emergency checkpoint at epoch {current_epoch}")
                self._save_checkpoint(model_ref, optimizer_ref, scheduler_ref, current_epoch)
                self.logger.success("✅ Emergency checkpoint saved!")
            self.logger.info("💡 You can resume training later with --resume latest")
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Training loop
        self.logger.section("TRAINING LOOP")
        
        for epoch in range(start_epoch, self.config.training.num_epochs + 1):
            self.current_epoch = epoch
            
            # Store variables for emergency checkpoint
            self.interrupt_save_vars = (model, optimizer, scheduler, epoch)
            
            # Check for interruption
            if self.interrupted:
                self.logger.info("🛑 Training stopped by user request")
                break
            
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.get_average_loss())
            else:
                scheduler.step()
            
            # Log metrics
            self._log_epoch_results(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_metrics.get_accuracy() > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics.get_accuracy()
                self._save_model(model, f"best_model_epoch_{epoch}.pth", val_metrics.get_accuracy())
                self.logger.success(f"🏆 New best model saved! Accuracy: {val_metrics.get_accuracy():.4f}")
            
            # Early stopping check
            if early_stopping(val_metrics.get_average_loss(), model):
                self.logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break
            
            # Save checkpoint every 5 epochs and on the last epoch
            if epoch % 5 == 0 or epoch == self.config.training.num_epochs:
                self._save_checkpoint(model, optimizer, scheduler, epoch)
        
        # Final evaluation on test set
        self.logger.section("FINAL EVALUATION")
        test_metrics = self._evaluate_model(model, test_loader, criterion)
        
        # Training summary
        training_time = time.time() - start_time
        
        results = {
            'best_val_accuracy': self.best_val_accuracy,
            'test_accuracy': test_metrics.get_accuracy(),
            'test_precision': test_metrics.get_precision(),
            'test_recall': test_metrics.get_recall(),
            'test_specificity': test_metrics.get_specificity(),
            'test_f1_score': test_metrics.get_f1_score(),
            'training_time': training_time,
            'total_epochs': self.current_epoch
        }
        
        self._log_final_results(results)
        return results
    
    def _train_epoch(self, model: nn.Module, train_loader, criterion, optimizer) -> CellexMetrics:
        """Train for one epoch."""
        model.train()
        metrics = CellexMetrics()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            metrics.update(outputs, targets, loss.item())
            
            # Log progress
            if batch_idx % self.config.logging.log_frequency == 0:
                self.logger.progress(
                    batch_idx, 
                    len(train_loader),
                    f"Training Epoch {self.current_epoch} - Loss: {loss.item():.4f}"
                )
        
        return metrics
    
    def _validate_epoch(self, model: nn.Module, val_loader, criterion) -> CellexMetrics:
        """Validate for one epoch."""
        model.eval()
        metrics = CellexMetrics()
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                metrics.update(outputs, targets, loss.item())
        
        return metrics
    
    def _evaluate_model(self, model: nn.Module, test_loader, criterion) -> CellexMetrics:
        """Comprehensive model evaluation."""
        self.logger.info("🧪 Running comprehensive evaluation...")
        
        model.eval()
        metrics = CellexMetrics()
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                metrics.update(outputs, targets, loss.item())
                
                if batch_idx % 50 == 0:
                    self.logger.progress(batch_idx, len(test_loader), "Evaluating")
        
        return metrics
    
    def _create_optimizer(self, model: nn.Module):
        """Create optimizer based on configuration."""
        if self.config.training.optimizer.lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            return optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        if self.config.training.scheduler == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.scheduler_params.get('T_max', self.config.training.num_epochs),
                eta_min=self.config.training.scheduler_params.get('eta_min', 1e-6)
            )
        elif self.config.training.scheduler == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config.training.scheduler == 'step':
            return StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def _log_epoch_results(self, epoch: int, train_metrics: CellexMetrics, 
                          val_metrics: CellexMetrics, learning_rate: float):
        """Log results for current epoch."""
        # Terminal logging
        self.logger.training_epoch(
            epoch,
            self.config.training.num_epochs,
            train_metrics.get_average_loss(),
            val_metrics.get_average_loss(),
            val_metrics.get_accuracy()
        )
        
        # Detailed metrics
        self.logger.info(f"📊 Train - Acc: {train_metrics.get_accuracy():.4f} | "
                        f"Prec: {train_metrics.get_precision():.4f} | "
                        f"Rec: {train_metrics.get_recall():.4f}")
        
        self.logger.info(f"📊 Val   - Acc: {val_metrics.get_accuracy():.4f} | "
                        f"Prec: {val_metrics.get_precision():.4f} | "
                        f"Rec: {val_metrics.get_recall():.4f} | "
                        f"F1: {val_metrics.get_f1_score():.4f}")
        
        # Update history
        self.training_history['train_loss'].append(train_metrics.get_average_loss())
        self.training_history['val_loss'].append(val_metrics.get_average_loss())
        self.training_history['train_acc'].append(train_metrics.get_accuracy())
        self.training_history['val_acc'].append(val_metrics.get_accuracy())
        self.training_history['learning_rate'].append(learning_rate)
        
        # MLflow logging
        if MLFLOW_AVAILABLE and self.config.logging.use_mlflow:
            try:
                mlflow.log_metrics({
                    'train_loss': train_metrics.get_average_loss(),
                    'val_loss': val_metrics.get_average_loss(),
                    'train_accuracy': train_metrics.get_accuracy(),
                    'val_accuracy': val_metrics.get_accuracy(),
                    'val_precision': val_metrics.get_precision(),
                    'val_recall': val_metrics.get_recall(),
                    'val_f1_score': val_metrics.get_f1_score(),
                    'learning_rate': learning_rate
                }, step=epoch)
            except:
                pass
        
        # Weights & Biases logging
        if WANDB_AVAILABLE and self.config.logging.use_wandb:
            try:
                wandb.log({
                    'train_loss': train_metrics.get_average_loss(),
                    'val_loss': val_metrics.get_average_loss(),
                    'train_accuracy': train_metrics.get_accuracy(),
                    'val_accuracy': val_metrics.get_accuracy(),
                    'val_precision': val_metrics.get_precision(),
                    'val_recall': val_metrics.get_recall(),
                    'val_f1_score': val_metrics.get_f1_score(),
                    'learning_rate': learning_rate
                }, step=epoch)
            except:
                pass
    
    def _save_model(self, model: nn.Module, filename: str, accuracy: float):
        """Save model with metadata."""
        model_path = self.model_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'epoch': self.current_epoch,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }, model_path)
    
    def _save_checkpoint(self, model: nn.Module, optimizer, scheduler, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history
        }, checkpoint_path)
        
        # Also save as "latest" for easy resuming
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history
        }, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer, scheduler=None):
        """Load checkpoint and resume training."""
        if not os.path.exists(checkpoint_path):
            # Try to find the checkpoint in the checkpoints directory
            if not checkpoint_path.startswith(str(self.checkpoint_dir)):
                alt_path = self.checkpoint_dir / checkpoint_path
                if alt_path.exists():
                    checkpoint_path = str(alt_path)
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if scheduler and checkpoint.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rate': []
            })
            
            self.logger.success(f"✅ Checkpoint loaded successfully!")
            self.logger.info(f"📊 Resuming from epoch {self.current_epoch + 1}")
            self.logger.info(f"🏆 Best validation accuracy so far: {self.best_val_accuracy:.2f}%")
            
            return self.current_epoch
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {str(e)}")
            raise
    
    def _log_final_results(self, results: Dict):
        """Log final training results."""
        self.logger.section("TRAINING COMPLETED")
        
        self.logger.metric("Best Validation Accuracy", results['best_val_accuracy'], "%")
        self.logger.metric("Test Accuracy", results['test_accuracy'], "%")
        self.logger.metric("Test Precision", results['test_precision'], "")
        self.logger.metric("Test Recall", results['test_recall'], "")
        self.logger.metric("Test Specificity", results['test_specificity'], "")
        self.logger.metric("Test F1 Score", results['test_f1_score'], "")
        self.logger.metric("Training Time", results['training_time'], "seconds")
        self.logger.metric("Total Epochs", results['total_epochs'], "")
        
        self.logger.success("🎉 Training completed successfully!")


def main():
    """Main training function."""
    config = get_config()
    trainer = CellexTrainer(config)
    
    # Check if processed data exists
    data_dir = Path(config.data.processed_data_dir) / "unified"
    
    if not data_dir.exists():
        trainer.logger.error("❌ Processed data not found!")
        trainer.logger.info("Please run data preprocessing first:")
        trainer.logger.info("python src/data/download_data.py")
        return
    
    # Start training
    results = trainer.train(str(data_dir))
    
    # Save results
    results_path = Path("results") / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    trainer.logger.success(f"✅ Results saved to {results_path}")


if __name__ == "__main__":
    main()