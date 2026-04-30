import torch
from torch.utils.data import DataLoader, TensorDataset
import config

def get_pytorch_loaders(X_train, X_test, y_train, y_test):
    # Convert to Tensors
    # LSTM expects 3D shape: (Batch, Sequence_Length, Features)
    # We use Sequence_Length = 1 for tabular data
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create PyTorch Datasets
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader