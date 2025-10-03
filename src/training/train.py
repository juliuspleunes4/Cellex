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
    
    def get_balanced_accuracy(self) -> float:
        """Calculate balanced accuracy (average of sensitivity and specificity)."""
        sensitivity = self.get_recall()
        specificity = self.get_specificity()
        return (sensitivity + specificity) / 2.0
    
    @property
    def avg_loss(self) -> float:
        """Calculate average loss."""
        return sum(self.losses) / len(self.losses) if self.losses else 0.0
    
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
        self.logger = get_logger("CellexTrainer", self.config.logging.log_file)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"[SYMBOL][INFO]  Device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"[SYMBOL] GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"[SYMBOL] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
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
                self.logger.success("[SUCCESS] MLflow initialized")
            except Exception as e:
                self.logger.warning(f"[WARNING]  MLflow initialization failed: {str(e)}")
        
        # Weights & Biases setup
        if WANDB_AVAILABLE and self.config.logging.use_wandb:
            try:
                wandb.init(
                    project=self.config.logging.experiment_name,
                    config=self.config.__dict__
                )
                self.logger.success("[SUCCESS] Weights & Biases initialized")
            except Exception as e:
                self.logger.warning(f"[WARNING]  Weights & Biases initialization failed: {str(e)}")
    
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
                self.logger.info(f"[PROGRESS] Resuming training from epoch {start_epoch}")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to resume from checkpoint: {str(e)}")
                self.logger.info("[PROGRESS] Starting fresh training instead...")
                start_epoch = 1
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            min_delta=0.001
        )
        
        # Optimize batch size if GPU memory allows
        if torch.cuda.is_available() and self.config.training.batch_size == 64:
            optimized_batch_size = self._optimize_batch_size(model, train_loader.dataset)
            if optimized_batch_size != self.config.training.batch_size:
                self.logger.info(f"[STATS] Optimizing batch size from {self.config.training.batch_size} to {optimized_batch_size}")
                # Recreate data loaders with optimized batch size
                train_loader, val_loader, test_loader = create_data_loaders(data_dir, self.config, batch_size=optimized_batch_size)
        
        # Calculate class weights if enabled
        if self.config.training.use_class_balancing and not self.config.training.class_weights:
            class_weights = self._calculate_class_weights(train_loader)
            self.config.training.class_weights = class_weights.tolist()
            self.logger.info(f"[STATS] Calculated class weights: {class_weights.tolist()}")
            
            # Recreate loss function with class weights
            criterion = create_loss_function(self.config)
            criterion = criterion.to(self.device)
        
        # Setup graceful shutdown for interruption
        def signal_handler(signum, frame):
            self.logger.info("\n[WARNING] Training interruption requested (Ctrl+C)")
            self.interrupted = True
            if self.interrupt_save_vars:
                model_ref, optimizer_ref, scheduler_ref, current_epoch = self.interrupt_save_vars
                self.logger.info(f"[SYMBOL] Saving emergency checkpoint at epoch {current_epoch}")
                self._save_checkpoint(model_ref, optimizer_ref, scheduler_ref, current_epoch)
                self.logger.success("[SUCCESS] Emergency checkpoint saved!")
            self.logger.info("[SYMBOL] You can resume training later with --resume latest")
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Training loop
        self.logger.section("TRAINING LOOP")
        
        for epoch in range(start_epoch, self.config.training.num_epochs + 1):
            self.current_epoch = epoch
            
            # Store variables for emergency checkpoint
            self.interrupt_save_vars = (model, optimizer, scheduler, epoch)
            
            # Check for interruption
            if self.interrupted:
                self.logger.info("[SYMBOL] Training stopped by user request")
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
            
            # Save best model and checkpoint (using balanced accuracy for imbalanced data)
            current_balanced_acc = val_metrics.get_balanced_accuracy()
            if current_balanced_acc > self.best_val_accuracy:
                self.best_val_accuracy = current_balanced_acc
                self._save_model(model, f"best_model_epoch_{epoch}.pth", current_balanced_acc)
                self._save_best_checkpoint(model, optimizer, scheduler, epoch)
                self.logger.success(f"[SYMBOL] New best model saved! Balanced Accuracy: {current_balanced_acc:.4f}")
            
            # Early stopping check
            if early_stopping(val_metrics.get_average_loss(), model):
                self.logger.info(f"[SYMBOL] Early stopping triggered at epoch {epoch}")
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
    
    def _optimize_batch_size(self, model: nn.Module, dataset) -> int:
        """Automatically optimize batch size based on GPU memory."""
        if not torch.cuda.is_available():
            return self.config.training.batch_size
        
        # Test different batch sizes to find optimal one
        test_batch_sizes = [128, 96, 64, 48, 32, 16]
        optimal_batch_size = self.config.training.batch_size
        
        model.train()
        dummy_input = torch.randn(1, 3, *self.config.data.image_size, device=self.device)
        
        for batch_size in test_batch_sizes:
            if batch_size <= self.config.training.batch_size:
                continue
                
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Test batch
                test_batch = dummy_input.expand(batch_size, -1, -1, -1)
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_batch)
                
                # Check if we have enough memory (keep 1GB free)
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                if memory_used < memory_total * 0.85:  # Use max 85% of GPU memory
                    optimal_batch_size = batch_size
                    self.logger.info(f"[STATS] GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return optimal_batch_size
    
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
            
            # Log progress - more frequent and informative
            if batch_idx % self.config.logging.log_frequency == 0:
                # Calculate current accuracy for real-time feedback
                current_acc = metrics.get_accuracy() if metrics.total > 0 else 0.0
                avg_loss = metrics.avg_loss if metrics.total > 0 else loss.item()
                
                # Create progress bar
                progress = batch_idx / len(train_loader)
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                
                # GPU memory info
                gpu_info = ""
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_percent = gpu_memory / gpu_total * 100
                    gpu_info = f"GPU: {gpu_memory:.1f}/{gpu_total:.1f}GB ({gpu_percent:.0f}%)"
                
                self.logger.info(
                    f"[PROGRESS] Epoch {self.current_epoch:2d} [{bar}] "
                    f"{batch_idx:3d}/{len(train_loader):3d} "
                    f"({100.0 * progress:5.1f}%) | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {current_acc:.2%}" + 
                    (f" | {gpu_info}" if gpu_info else "")
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
        self.logger.info("[TEST] Running comprehensive evaluation...")
        
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
    
    def _calculate_class_weights(self, train_loader):
        """Calculate class weights for balanced training."""
        class_counts = torch.zeros(2)  # Assuming binary classification
        total_samples = 0
        
        self.logger.info("[STATS] Calculating class distribution...")
        
        for batch_idx, (_, targets) in enumerate(train_loader):
            for target in targets:
                class_counts[target] += 1
                total_samples += 1
            
            # Show progress for large datasets
            if batch_idx % 100 == 0:
                self.logger.info(f"[PROGRESS] Processed {batch_idx * train_loader.batch_size:,} samples...")
        
        # Calculate weights (inverse frequency)
        class_weights = total_samples / (2.0 * class_counts)
        
        self.logger.info(f"[STATS] Class distribution: Healthy={int(class_counts[0])}, Cancer={int(class_counts[1])}")
        self.logger.info(f"[STATS] Class percentages: Healthy={100*class_counts[0]/total_samples:.1f}%, Cancer={100*class_counts[1]/total_samples:.1f}%")
        
        return class_weights
    
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
        self.logger.info(f"[STATS] Train - Acc: {train_metrics.get_accuracy():.4f} | "
                        f"Prec: {train_metrics.get_precision():.4f} | "
                        f"Rec: {train_metrics.get_recall():.4f}")
        
        self.logger.info(f"[STATS] Val   - Acc: {val_metrics.get_accuracy():.4f} | "
                        f"Bal.Acc: {val_metrics.get_balanced_accuracy():.4f} | "
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
                    'val_balanced_accuracy': val_metrics.get_balanced_accuracy(),
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
    
    def _save_best_checkpoint(self, model: nn.Module, optimizer, scheduler, epoch: int):
        """Save best model checkpoint for resuming training from best state."""
        best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history
        }, best_checkpoint_path)
    
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
        
        self.logger.info(f"[FILE] Loading checkpoint from: {checkpoint_path}")
        
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
            
            self.logger.success(f"[SUCCESS] Checkpoint loaded successfully!")
            self.logger.info(f"[STATS] Resuming from epoch {self.current_epoch + 1}")
            self.logger.info(f"[SYMBOL] Best validation accuracy so far: {self.best_val_accuracy:.2f}%")
            
            return self.current_epoch
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load checkpoint: {str(e)}")
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
        
        self.logger.success("[COMPLETE] Training completed successfully!")


def main():
    """Main training function."""
    config = get_config()
    trainer = CellexTrainer(config)
    
    # Check if processed data exists
    data_dir = Path(config.data.processed_data_dir) / "unified"
    
    if not data_dir.exists():
        trainer.logger.error("[ERROR] Processed data not found!")
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
    
    trainer.logger.success(f"[SUCCESS] Results saved to {results_path}")


if __name__ == "__main__":
    main()