"""
Training script for Cellex CNN model
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import create_model
from dataset import create_data_loaders, generate_synthetic_dataset


class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
    
    def update_train(self, loss, accuracy):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
    
    def update_val(self, loss, accuracy, precision, recall, f1):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1_scores.append(f1)
    
    def get_summary(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1_scores': self.val_f1_scores
        }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return epoch_loss, accuracy * 100, precision, recall, f1, cm


def train_model(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Generate or load dataset
    print("\nPreparing dataset...")
    if args.use_synthetic:
        print("Generating synthetic dataset...")
        image_paths, labels = generate_synthetic_dataset(
            args.data_dir, 
            num_samples_per_class=args.num_samples
        )
    else:
        # Load real dataset (implement based on your data structure)
        raise NotImplementedError("Real dataset loading not implemented. Use --use-synthetic flag.")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, cm = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update metrics
        metrics_tracker.update_train(train_loss, train_acc)
        metrics_tracker.update_val(val_loss, val_acc, val_precision, val_recall, val_f1)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, 'checkpoints', f'model_epoch_{epoch + 1}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'checkpoints', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Save training metrics
    metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_tracker.get_summary(), f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {args.output_dir}/checkpoints/")
    print(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Cellex CNN for X-ray classification')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='ml_model/datasets/synthetic',
                        help='Path to dataset directory')
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                        help='Use synthetic dataset for demonstration')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of samples per class for synthetic data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='ml_model/models',
                        help='Directory to save models and metrics')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
