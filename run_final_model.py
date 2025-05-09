import torch
from torch.utils.data import DataLoader
import time

# Import your data loader
from data_loader import ChunkedHiggsDataset, create_data_loaders

# Import models and training functions
from higgs_model import (
    Baseline, DeeperNN, 
    train_model, evaluate_model, 
    plot_training_history, plot_curves
)

def main():
    # Configuration
    file_path = '/Users/tribrid/Desktop/Machine Learning/Higgs-Classification/HIGGS.csv'
    epochs = 10
    batch_size = 4096
    lr = 0.001
    weight_decay = 1e-5
    chunk_size = 200000
    max_chunks = None
    use_class_weighting = True
    weight_factor = 0.5
    find_best_threshold = True
    
    # Print experiment configuration
    print(f"\nRunning DeeperNN with High-Level Features")
    print(f"Dataset: {file_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Max chunks: {max_chunks if max_chunks else 'All'}")
    print(f"Use class weighting: {use_class_weighting}")
    print(f"Weight factor: {weight_factor}")
    print(f"Find best threshold: {find_best_threshold}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Timing the experiment
    start_time = time.time()
    
    # Create data loaders for high-level features only
    print("\nCreating data loaders for high-level features...")
    train_loader, val_loader, test_loader, num_features = create_data_loaders(
        file_path=file_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        use_low_level=False,  # Only use high-level features
        use_high_level=True
    )
    
    print(f"\nNumber of high-level features: {num_features}")
    
    # Create DeeperNN model
    print("\nCreating DeeperNN model...")
    model = DeeperNN(input_dim=num_features, hidden_dim=128)
    print(model)
    
    # Train model
    print("\nTraining DeeperNN with high-level features...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_chunks=max_chunks,
        device=device,
        use_class_weighting=use_class_weighting,
        weight_factor=weight_factor
    )
    
    # Evaluate model with threshold tuning
    print("\nEvaluating model on test set with threshold tuning...")
    metrics = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        val_loader=val_loader if find_best_threshold else None,
        max_chunks=max_chunks,
        device=device,
        find_best_threshold=find_best_threshold
    )
    
    # Plot results
    plot_training_history(history, title='DeeperNN with High-Level Features')
    plot_curves(metrics, title='DeeperNN Evaluation')
    
    # Print total experiment time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()