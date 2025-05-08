import torch
from torch.utils.data import DataLoader
import time
from data_loader import ChunkedHiggsDataset, create_data_loaders
from higgs_model import (
    Baseline, DeeperNN, 
    train_model, evaluate_model, 
    plot_training_history, plot_curves,
    compare_feature_sets
)

def main():
    # Configuration
    file_path = '/Users/tribrid/Desktop/Machine Learning/Higgs-Classification/HIGGS.csv'
    epochs = 10
    batch_size = 4096
    lr = 0.001
    weight_decay = 1e-5
    chunk_size = 200000
    # Set to None to use all data
    max_chunks = 15
    # Set to True to compare different feature sets
    compare_features = True  
    save_models = True
    
    # Print experiment configuration
    print(f"\nRunning Higgs Boson Classification Baseline Experiment")
    print(f"Dataset: {file_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Max chunks: {max_chunks if max_chunks else 'All'}")
    print(f"Feature set comparison: {compare_features}")
    print(f"Save models: {save_models}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Timing the entire experiment
    start_time = time.time()
    
    if compare_features:
        # Create data loaders for different feature sets
        print("\nCreating data loaders for different feature sets...")
        
        # All features
        print("\nCreating data loaders for all features...")
        train_loader_all, val_loader_all, test_loader_all, num_features_all = create_data_loaders(
            file_path=file_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            use_low_level=True,
            use_high_level=True
        )
        
        # Low-level features only
        print("\nCreating data loaders for low-level features only...")
        train_loader_low, val_loader_low, test_loader_low, num_features_low = create_data_loaders(
            file_path=file_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            use_low_level=True,
            use_high_level=False
        )
        
        # High-level features only
        print("\nCreating data loaders for high-level features only...")
        train_loader_high, val_loader_high, test_loader_high, num_features_high = create_data_loaders(
            file_path=file_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            use_low_level=False,
            use_high_level=True
        )
        
        # Create dictionaries for compare_feature_sets function
        train_loaders = {
            'All Features': train_loader_all,
            'Low-Level Features': train_loader_low,
            'High-Level Features': train_loader_high
        }
        
        val_loaders = {
            'All Features': val_loader_all,
            'Low-Level Features': val_loader_low,
            'High-Level Features': val_loader_high
        }
        
        test_loaders = {
            'All Features': test_loader_all,
            'Low-Level Features': test_loader_low,
            'High-Level Features': test_loader_high
        }
        
        input_dims = {
            'All Features': num_features_all,
            'Low-Level Features': num_features_low,
            'High-Level Features': num_features_high
        }
        
        # Compare feature sets
        results = compare_feature_sets(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            input_dims=input_dims,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_chunks=max_chunks
        )
        
    else:
        # Just train a single model with all features
        print("\nCreating data loaders for all features...")
        train_loader, val_loader, test_loader, num_features = create_data_loaders(
            file_path=file_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            use_low_level=True,
            use_high_level=True
        )
        
        print(f"\nNumber of features: {num_features}")
        
        # Creating baseline model
        model = Baseline(input_dim=num_features)
        
        # Training the baseline model
        print("\nTraining baseline model...")
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_chunks=max_chunks,
            device=device
        )
        
        # Evaluate the baseline model on the test set
        print("\nEvaluating model on test set...")
        metrics = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            max_chunks=max_chunks,
            device=device
        )
        
        # Plot results
        plot_training_history(history, title='Baseline Model Training History')
        plot_curves(metrics, title='Baseline Model Evaluation')
        
        # Save model
        if save_models:
            torch.save(trained_model.state_dict(), "higgs_baseline_model.pt")
            print("Model saved as 'higgs_baseline_model.pt'")
    
    # Print total experiment time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()