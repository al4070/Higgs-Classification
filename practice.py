# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix


# # Baseline model
# class SimpleNN(nn.Module):
#     def __init__(self, input_dim=28, hidden_dim=64):
#         super(SimpleNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         return self.model(x)


# model = SimpleNN(input_dim=28).to(device)

# # Training parameters
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# criterion = nn.BCELoss()
# epochs = 5  # Start with fewer epochs to establish baseline

# # Track metrics
# best_val_auc = 0

# # For quick validation, let the data loader terminate after a few chunks
# max_chunks = 5  # For testing, only ~1 million examples