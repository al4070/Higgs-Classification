import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix,
    roc_auc_score  
)

# Baseline model
class Baseline(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64):
        super(Baseline, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Deeper model with batch normalization
class DeeperNN(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128):
        super(DeeperNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, weight_decay=1e-5, max_chunks=None, device=None):
    """
    Train a PyTorch model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train for
        lr: Learning rate
        weight_decay: L2 regularization weight
        max_chunks: Maximum number of chunks to process (for testing)
        device: Device to train on (CPU or GPU)
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'val_ap': [],
        'epoch_times': []
    }
    
    # Track best validation AUC for model selection
    best_val_auc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Training metrics
        train_loss = 0.0
        train_samples = 0
        chunk_count = 0
        
        # Process batches
        for features, labels in train_loader:
            # Check if we've processed enough chunks (for testing)
            chunk_count += 1
            if max_chunks is not None and chunk_count > max_chunks:
                break
                
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * labels.size(0)
            train_samples += labels.size(0)
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_labels = []
        chunk_count = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                # Check if we've processed enough chunks
                chunk_count += 1
                if max_chunks is not None and chunk_count > max_chunks:
                    break
                    
                # Move data to device
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item() * labels.size(0)
                val_samples += labels.size(0)
                
                # Store predictions and labels for computing AUC, etc.
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        history['val_loss'].append(avg_val_loss)
        
        # Binary predictions for accuracy
        bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        val_accuracy = np.mean(np.array(bin_preds) == np.array(all_labels))
        history['val_accuracy'].append(val_accuracy)
        
        # Calculate AUC and AP if we have both classes
        unique_labels = np.unique(all_labels)
        if len(unique_labels) > 1:
            val_auc = roc_auc_score(all_labels, all_preds)
            val_ap = average_precision_score(all_labels, all_preds)
        else:
            # Default values if only one class is present
            val_auc = 0.5
            val_ap = np.mean(all_labels)
        
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Val AP: {val_ap:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, max_chunks=None, device=None):
    """
    Evaluate a trained model on a test set
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        max_chunks: Maximum number of chunks to process (for testing)
        device: Device to evaluate on (CPU or GPU)
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    chunk_count = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            # Check if we've processed enough chunks
            chunk_count += 1
            if max_chunks is not None and chunk_count > max_chunks:
                break
                
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features).squeeze()
            
            # Store predictions and labels
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Binary predictions for accuracy and confusion matrix
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    
    # Compute metrics
    accuracy = np.mean(np.array(bin_preds) == np.array(all_labels))
    
    # Calculate AUC and AP if we have both classes
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        test_auc = roc_auc_score(all_labels, all_preds)
        test_ap = average_precision_score(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        cm = confusion_matrix(all_labels, bin_preds)
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    else:
        # Default values if only one class is present
        test_auc = 0.5
        test_ap = np.mean(all_labels)
        fpr, tpr = [0, 1], [0, 1]
        precision, recall = [0, 1], [0, 1]
        if all_labels[0] == 1:
            cm = np.array([[0, 0], [0, len(all_labels)]])
            specificity = 0
            sensitivity = 1
            f1 = 0
        else:
            cm = np.array([[len(all_labels), 0], [0, 0]])
            specificity = 1
            sensitivity = 0
            f1 = 0
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'auc': test_auc,
        'ap': test_ap,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision, recall)
    }
    
    # Print metrics
    print("\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
            print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    return metrics

# Function to visualize training history
def plot_training_history(history, title='Model Training History'):
    """
    Plot training metrics
    
    Args:
        history: Dictionary containing training history
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # Plot AP
    plt.subplot(2, 2, 4)
    plt.plot(history['val_ap'], label='Validation AP')
    plt.title('Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save and show plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Function to plot ROC and PR curves
def plot_curves(metrics, title='Model Evaluation'):
    """
    Plot ROC and PR curves
    
    Args:
        metrics: Dictionary containing evaluation metrics
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Plot ROC curve
    plt.subplot(1, 2, 1)
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot PR curve
    plt.subplot(1, 2, 2)
    precision, recall = metrics['pr_curve']
    plt.plot(recall, precision, label=f"AP = {metrics['ap']:.4f}")
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Save and show plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Function to run experiments with multiple feature sets
def compare_feature_sets(train_loaders, val_loaders, test_loaders, input_dims, epochs=5, lr=0.001, weight_decay=1e-5, max_chunks=None):
    """
    Compare models with different feature sets
    
    Args:
        train_loaders: Dictionary of training data loaders for each feature set
        val_loaders: Dictionary of validation data loaders for each feature set
        test_loaders: Dictionary of test data loaders for each feature set
        input_dims: Dictionary of input dimensions for each feature set
        epochs: Number of epochs to train for
        lr: Learning rate
        weight_decay: L2 regularization weight
        max_chunks: Maximum number of chunks to process (for testing)
        
    Returns:
        results: Dictionary containing results for each feature set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for feature_set, train_loader in train_loaders.items():
        print(f"\n{'='*50}")
        print(f"Training model with {feature_set}...")
        print(f"{'='*50}")
        
        # Create model
        input_dim = input_dims[feature_set]
        model = Baseline(input_dim=input_dim)
        
        # Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loaders[feature_set],
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_chunks=max_chunks,
            device=device
        )
        
        # Evaluate model
        metrics = evaluate_model(
            model=trained_model,
            test_loader=test_loaders[feature_set],
            max_chunks=max_chunks,
            device=device
        )
        
        # Plot results
        plot_training_history(history, title=f'Training History - {feature_set}')
        plot_curves(metrics, title=f'Evaluation Curves - {feature_set}')
        
        # Save model
        torch.save(trained_model.state_dict(), f"higgs_model_{feature_set.lower().replace(' ', '_')}.pt")
        
        # Store results
        results[feature_set] = {
            'model': trained_model,
            'history': history,
            'metrics': metrics
        }
    
    # Print comparison table
    print("\n")
    print("="*80)
    print("FEATURE SET COMPARISON")
    print("="*80)
    print(f"{'Feature Set':<20} {'Accuracy':<10} {'AUC':<10} {'AP':<10} {'F1':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-"*80)
    
    for feature_set, result in results.items():
        metrics = result['metrics']
        print(f"{feature_set:<20} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['auc']:<10.4f} "
              f"{metrics['ap']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f}")
    
    return results

# Example usage:
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple model for demonstration
    model = Baseline(input_dim=28).to(device)
    print(model)