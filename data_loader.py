import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class ChunkedHiggsDataset(IterableDataset):
    """
    Memory-efficient dataset for large CSV files (8GB+)
    Uses chunking to load and process data in smaller pieces
    """
    def __init__(self, file_path, chunk_size=100000, transform=None, 
                 use_low_level=True, use_high_level=True, shuffle=True):
        """
        Initialize the dataset
        
        Args:
            file_path: Path to the CSV file
            chunk_size: Number of rows to load at once
            transform: Optional transform to apply to features
            use_low_level: Whether to use low-level features
            use_high_level: Whether to use high-level features
            shuffle: Whether to shuffle chunks
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.transform = transform
        self.use_low_level = use_low_level
        self.use_high_level = use_high_level
        self.shuffle = shuffle
        
        # Column names for the dataset
        self.column_names = [
            "label",
            # Low-level features (1–21)
            "lepton_pT", "lepton_eta", "lepton_phi",
            "missing_energy_magnitude", "missing_energy_phi",
            "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
            "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
            "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
            "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
            # High-level features (22–28)
            "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
        ]
        
        # Get feature column indices
        self.feature_cols = []
        if use_low_level:
            self.feature_cols.extend(list(range(1, 22)))  # Low-level features (columns 1-21)
        if use_high_level:
            self.feature_cols.extend(list(range(22, 29)))  # High-level features (columns 22-28)
        
        # Calculate total number of rows in the file
        # Count the number of lines in the file efficiently
        with open(file_path, 'r') as f:
            self.total_rows = sum(1 for _ in f) - 1  # Subtract header
        
        # Calculate number of chunks
        self.num_chunks = int(np.ceil(self.total_rows / self.chunk_size))
        
        # Calculate offsets for each chunk
        self.offsets = list(range(0, self.total_rows, self.chunk_size))
        if self.shuffle:
            np.random.shuffle(self.offsets)
    
    def __iter__(self):
        """
        Iterator for the dataset
        Yields batches of (features, labels) as tensors
        """
        # Iterate through chunks
        for offset in self.offsets:
            # Skip rows to reach the offset and read a chunk
            df_chunk = pd.read_csv(
                self.file_path, 
                header=None, 
                names=self.column_names,
                skiprows=1+offset,  # +1 for header
                nrows=self.chunk_size
            )
            
            # Extract labels
            labels = df_chunk['label'].values
            
            # Extract features based on which feature set is selected
            if self.use_low_level and self.use_high_level:
                features = df_chunk.iloc[:, 1:].values
            elif self.use_low_level:
                features = df_chunk.iloc[:, 1:22].values
            elif self.use_high_level:
                features = df_chunk.iloc[:, 22:29].values
            
            # Apply transform if provided
            if self.transform:
                features = self.transform(features)
            
            # Convert to torch tensors
            features_tensor = torch.FloatTensor(features)
            labels_tensor = torch.FloatTensor(labels)
            
            # Yield individual samples
            for i in range(len(labels_tensor)):
                yield features_tensor[i], labels_tensor[i]

def create_data_loaders(file_path, chunk_size=100000, batch_size=1024, 
                        use_low_level=True, use_high_level=True,
                        train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train, validation, and test data loaders for large datasets
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows to process at once
        batch_size: Batch size for the data loaders
        use_low_level: Whether to use low-level features
        use_high_level: Whether to use high-level features
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, num_features
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Estimate number of rows in the file
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Estimate based on first line length
    est_rows = int(file_size / len(first_line))
    print(f"Estimated rows in dataset: {est_rows:,}")
    
    # Calculate approximate sizes for each split
    total_chunks = int(np.ceil(est_rows / chunk_size))
    train_chunks = int(total_chunks * train_ratio)
    val_chunks = int(total_chunks * val_ratio)
    test_chunks = total_chunks - train_chunks - val_chunks
    
    print(f"Splitting into approximately:")
    print(f"  Training: {train_chunks} chunks (~{train_chunks * chunk_size:,} samples)")
    print(f"  Validation: {val_chunks} chunks (~{val_chunks * chunk_size:,} samples)")
    print(f"  Testing: {test_chunks} chunks (~{test_chunks * chunk_size:,} samples)")
    
    # Create a scaler with a small sample for feature normalization
    print("Creating feature scaler...")
    sample_df = pd.read_csv(file_path, header=None, nrows=200000)
    sample_df.columns = ['label'] + [f'feature_{i}' for i in range(1, 29)]
    
    # Select features
    feature_cols = []
    if use_low_level:
        feature_cols.extend(list(range(1, 22)))
    if use_high_level:
        feature_cols.extend(list(range(22, 29)))
    
    # Fit scaler on selected features
    scaler = StandardScaler()
    features_sample = sample_df.iloc[:, feature_cols].values
    scaler.fit(features_sample)
    
    transform = lambda x: scaler.transform(x)
    
    # Create offsets for the full file
    all_offsets = np.arange(0, est_rows, chunk_size)
    np.random.shuffle(all_offsets)
    
    # Split offsets for train/val/test
    train_offsets = all_offsets[:train_chunks]
    val_offsets = all_offsets[train_chunks:train_chunks+val_chunks]
    test_offsets = all_offsets[train_chunks+val_chunks:]
    
    # Custom datasets for each split
    class TrainDataset(ChunkedHiggsDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.offsets = train_offsets
    
    class ValDataset(ChunkedHiggsDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.offsets = val_offsets
    
    class TestDataset(ChunkedHiggsDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.offsets = test_offsets
    
    # Create datasets
    train_dataset = TrainDataset(
        file_path=file_path,
        chunk_size=chunk_size,
        transform=transform,
        use_low_level=use_low_level,
        use_high_level=use_high_level,
        shuffle=True
    )
    
    val_dataset = ValDataset(
        file_path=file_path,
        chunk_size=chunk_size,
        transform=transform,
        use_low_level=use_low_level,
        use_high_level=use_high_level,
        shuffle=False
    )
    
    test_dataset = TestDataset(
        file_path=file_path,
        chunk_size=chunk_size,
        transform=transform,
        use_low_level=use_low_level,
        use_high_level=use_high_level,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # Adjust based on available CPUs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    
    # Calculate and return number of features
    num_features = 0
    if use_low_level:
        num_features += 21
    if use_high_level:
        num_features += 7
    
    return train_loader, val_loader, test_loader, num_features

# Example usage for testing the data loader
if __name__ == "__main__":
    file_path = '/Users/tribrid/Desktop/Machine Learning/Higgs-Classification/HIGGS.csv'
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_features = create_data_loaders(
        file_path=file_path,
        chunk_size=200000,
        batch_size=4096,
        use_low_level=True,
        use_high_level=True
    )
    
    print(f"Number of features: {num_features}")
    
    # Sample batch from training loader
    for batch_features, batch_labels in train_loader:
        print(f"Batch shape: {batch_features.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Class balance in batch: {torch.mean(batch_labels).item():.4f}")
        break  # Just check one batch