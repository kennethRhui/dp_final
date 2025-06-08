from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch

# Global settings
num_clients = 5  # Number of clients
noniid_classes_per_client = 2  # Number of classes per client (Non-IID setting)

def get_dataloaders(client_id, batch_size=32, train_ratio=0.8):
    """
    Fixed data loader - ensure each client has test data
    """
    global num_clients, noniid_classes_per_client
    
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    targets = np.array(mnist.targets)
    indices_per_class = {i: np.where(targets == i)[0].tolist() for i in range(10)}

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(seed=42)

    # Allocate data to each client (Non-IID approach)
    for cid in range(num_clients):
        chosen_classes = rng.choice(range(10), size=noniid_classes_per_client, replace=False)
        for cls in chosen_classes:
            if len(indices_per_class[cls]) >= 200:  # Ensure sufficient samples
                chosen_idx = rng.choice(indices_per_class[cls], size=200, replace=False).tolist()
                client_indices[cid].extend(chosen_idx)
                # Remove selected indices to avoid duplication
                for idx in chosen_idx:
                    indices_per_class[cls].remove(idx)

    # Get data for specified client
    if client_id >= num_clients:
        print(f"Error: client_id {client_id} exceeds num_clients {num_clients}")
        return None
        
    client_data = Subset(mnist, client_indices[client_id])
    
    # Check data amount
    total_samples = len(client_data)
    print(f"Client {client_id}: Total samples = {total_samples}")
    
    if total_samples == 0:
        print(f"Error: Client {client_id} has no data")
        return None

    # Ensure minimum training and test sample counts
    min_train_samples = 50  # At least 50 training samples per client
    min_test_samples = 20   # At least 20 test samples per client
    
    # If data is insufficient, adjust allocation
    if total_samples < min_train_samples + min_test_samples:
        print(f"Warning: Client {client_id} has insufficient data ({total_samples} samples)")
        # Adjust minimum requirements
        min_train_samples = max(10, int(total_samples * 0.7))
        min_test_samples = total_samples - min_train_samples
    
    # Split data
    train_size = max(min_train_samples, int(total_samples * train_ratio))
    test_size = total_samples - train_size
    
    # Ensure test data is not empty
    if test_size < min_test_samples and total_samples > min_test_samples:
        test_size = min_test_samples
        train_size = total_samples - test_size
    
    # Final check
    if test_size <= 0:
        test_size = max(1, total_samples // 5)  # At least 1 test sample, or 20% of total
        train_size = total_samples - test_size
    
    print(f"Client {client_id}: Train={train_size}, Test={test_size}")
    
    try:
        train_data, test_data = torch.utils.data.random_split(
            client_data, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # Fixed random seed
        )
    except ValueError as e:
        print(f"Error splitting data for client {client_id}: {e}")
        return None
    
    # Ensure test data is not empty
    if len(test_data) == 0:
        print(f"Warning: No test data for client {client_id}")
        return None
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def check_data_distribution():
    """Helper function to check data distribution"""
    print("Checking data distribution...")
    for client_id in range(num_clients):
        result = get_dataloaders(client_id)
        if result is None:
            print(f"Client {client_id}: No data available")
        else:
            train_loader, test_loader = result
            train_size = len(train_loader.dataset) if train_loader else 0
            test_size = len(test_loader.dataset) if test_loader else 0
            total_size = train_size + test_size
            print(f"Client {client_id}: Total={total_size}, Train={train_size}, Test={test_size}")

if __name__ == "__main__":
    # Test data distribution
    check_data_distribution()
