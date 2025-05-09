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
            #Instead of havung a sigmoid BCEWithLogitsLoss has an internal sigmoid
        )
    
    def forward(self, x):
        return self.model(x)

# This will be the final model
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
        )
    
    def forward(self, x):
        return self.model(x)


# Function to calculate class weights from a loader with weight factor
def calculate_class_weights(loader, max_chunks=None, device=None, weight_factor=0.5):
    """
    Calculate class weights based on class distribution in the dataset
    
    Args:
        loader: DataLoader to sample from
        max_chunks: Maximum number of chunks to process
        device: Computation device
        weight_factor: Factor to moderate the class weighting (0.0-1.0)
                       Lower values reduce the effect of class weighting
        
    Returns:
        pos_weight: Weight for positive class in BCEWithLogitsLoss
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos_count = 0
    neg_count = 0
    chunk_count = 0
    
    # Sample from the loader to estimate class distribution
    for _, labels in loader:
        chunk_count += 1
        if max_chunks is not None and chunk_count > max_chunks:
            break
        
        pos_count += (labels == 1).sum().item()
        neg_count += (labels == 0).sum().item()
    
    print(f"Class distribution - Positive: {pos_count}, Negative: {neg_count}")
    
    # If no examples of a class, use a balanced weight
    if pos_count == 0 or neg_count == 0:
        print("Warning: One class has zero samples in estimation set. Using balanced weighting.")
        pos_weight = torch.tensor([1.0], device=device)
    else:
        # Calculate pos_weight for BCEWithLogitsLoss
        # Multiply by weight_factor to moderate the weighting
        raw_weight = neg_count/pos_count
        pos_weight = torch.tensor([weight_factor * raw_weight], device=device)
    
    print(f"Raw weight ratio (neg/pos): {neg_count/pos_count:.4f}")
    print(f"Using weight factor: {weight_factor}")
    print(f"Final positive class weight: {pos_weight.item():.4f}")
    return pos_weight

# Modified training function to accept weight_factor
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, weight_decay=1e-5, 
                max_chunks=None, device=None, use_class_weighting=True, weight_factor=0.5):
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
        use_class_weighting: Whether to use class weighting for imbalanced data
        weight_factor: Factor to moderate the class weighting (0.0-1.0)
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use class weighting if specified
    if use_class_weighting:
        print("Calculating class weights...")
        pos_weight = calculate_class_weights(train_loader, max_chunks=max_chunks, 
                                            device=device, weight_factor=weight_factor)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()  # No class weighting
    
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
                # We need to apply sigmoid here since our model doesn't have it
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
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


# Evaluation function with threshold tuning
def evaluate_model(model, test_loader, val_loader=None, max_chunks=None, device=None, find_best_threshold=True):
    """
    Evaluate a trained model on a test set with threshold tuning
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        val_loader: Validation data loader (used to find optimal threshold)
        max_chunks: Maximum number of chunks to process (for testing)
        device: Device to evaluate on (CPU or GPU)
        find_best_threshold: Whether to find the optimal threshold on validation set
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Find optimal threshold on validation set if requested
    optimal_threshold = 0.5  # Default threshold
    
    if find_best_threshold and val_loader is not None:
        print("Finding optimal threshold on validation set...")
        # Collect validation predictions
        val_preds = []
        val_labels = []
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
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                
                # Store predictions and labels
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Try different thresholds to find optimal F1 score
        thresholds = np.linspace(0.1, 0.9, 30)  # Try 30 thresholds between 0.1 and 0.9
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            # Make binary predictions using this threshold
            binary_preds = [1 if p >= threshold else 0 for p in val_preds]
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            f1 = f1_score(val_labels, binary_preds)
            
            # Check if this is the best F1 score so far
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_threshold = best_threshold
        print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    # Evaluate on test set using optimal threshold
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
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Store predictions and labels
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Binary predictions using optimal threshold
    bin_preds = [1 if p >= optimal_threshold else 0 for p in all_preds]
    
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
        'pr_curve': (precision, recall),
        'threshold': optimal_threshold
    }
    
    # Print metrics
    print("\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name not in ['confusion_matrix', 'roc_curve', 'pr_curve', 'threshold']:
            print(f"{metric_name}: {metric_value:.4f}")
    
    print(f"Threshold: {optimal_threshold:.4f}")
    
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


# Function to run experiments with multiple feature sets with threshold tuning
def compare_feature_sets(train_loaders, val_loaders, test_loaders, input_dims, epochs=5, 
                         lr=0.001, weight_decay=1e-5, max_chunks=None, use_class_weighting=True,
                         weight_factor=0.5, find_best_threshold=True):
    """
    Compare models with different feature sets with threshold tuning
    
    Args:
        train_loaders: Dictionary of training data loaders for each feature set
        val_loaders: Dictionary of validation data loaders for each feature set
        test_loaders: Dictionary of test data loaders for each feature set
        input_dims: Dictionary of input dimensions for each feature set
        epochs: Number of epochs to train for
        lr: Learning rate
        weight_decay: L2 regularization weight
        max_chunks: Maximum number of chunks to process (for testing)
        use_class_weighting: Whether to use class weighting for imbalanced data
        weight_factor: Factor to moderate the class weighting (0.0-1.0)
        find_best_threshold: Whether to find the optimal threshold
        
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
            device=device,
            use_class_weighting=use_class_weighting,
            weight_factor=weight_factor
        )
        
        # Evaluate model with threshold tuning
        metrics = evaluate_model(
            model=trained_model,
            test_loader=test_loaders[feature_set],
            val_loader=val_loaders[feature_set] if find_best_threshold else None,
            max_chunks=max_chunks,
            device=device,
            find_best_threshold=find_best_threshold
        )
        
    
        # Store results
        results[feature_set] = {
            'history': history,
            'metrics': metrics
        }
    
    # Print comparison table
    print("\n")
    print("="*80)
    print("FEATURE SET COMPARISON")
    print("="*80)
    print(f"{'Feature Set':<20} {'Accuracy':<10} {'AUC':<10} {'AP':<10} {'F1':<10} {'Sensitivity':<12} {'Specificity':<12} {'Threshold':<10}")
    print("-"*100)
    
    for feature_set, result in results.items():
        metrics = result['metrics']
        print(f"{feature_set:<20} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['auc']:<10.4f} "
              f"{metrics['ap']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} "
              f"{metrics['threshold']:<10.4f}")
    
    return results










if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple model for demonstration
    model = Baseline(input_dim=28).to(device)
    print(model)