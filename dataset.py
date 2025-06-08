from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch

# 全域設定
num_clients = 5  # 客戶端數量
noniid_classes_per_client = 2  # 每個客戶端的類別數（Non-IID設定）

def get_dataloaders(client_id, batch_size=32, train_ratio=0.8):
    """
    修正的資料載入器 - 確保每個客戶端都有測試數據
    """
    global num_clients, noniid_classes_per_client
    
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    targets = np.array(mnist.targets)
    indices_per_class = {i: np.where(targets == i)[0].tolist() for i in range(10)}

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(seed=42)

    # 為每個客戶端分配數據（Non-IID方式）
    for cid in range(num_clients):
        chosen_classes = rng.choice(range(10), size=noniid_classes_per_client, replace=False)
        for cls in chosen_classes:
            if len(indices_per_class[cls]) >= 200:  # 確保有足夠的樣本
                chosen_idx = rng.choice(indices_per_class[cls], size=200, replace=False).tolist()
                client_indices[cid].extend(chosen_idx)
                # 移除已選擇的索引，避免重複
                for idx in chosen_idx:
                    indices_per_class[cls].remove(idx)

    # 獲取指定客戶端的數據
    if client_id >= num_clients:
        print(f"Error: client_id {client_id} exceeds num_clients {num_clients}")
        return None
        
    client_data = Subset(mnist, client_indices[client_id])
    
    # 檢查數據量
    total_samples = len(client_data)
    print(f"Client {client_id}: Total samples = {total_samples}")
    
    if total_samples == 0:
        print(f"Error: Client {client_id} has no data")
        return None

    # 確保最小的訓練和測試樣本數
    min_train_samples = 50  # 每個客戶端至少50個訓練樣本
    min_test_samples = 20   # 每個客戶端至少20個測試樣本
    
    # 如果數據太少，調整分配
    if total_samples < min_train_samples + min_test_samples:
        print(f"Warning: Client {client_id} has insufficient data ({total_samples} samples)")
        # 調整最小需求
        min_train_samples = max(10, int(total_samples * 0.7))
        min_test_samples = total_samples - min_train_samples
    
    # 分割數據
    train_size = max(min_train_samples, int(total_samples * train_ratio))
    test_size = total_samples - train_size
    
    # 確保測試數據不為空
    if test_size < min_test_samples and total_samples > min_test_samples:
        test_size = min_test_samples
        train_size = total_samples - test_size
    
    # 最終檢查
    if test_size <= 0:
        test_size = max(1, total_samples // 5)  # 至少1個測試樣本，或總數的20%
        train_size = total_samples - test_size
    
    print(f"Client {client_id}: Train={train_size}, Test={test_size}")
    
    try:
        train_data, test_data = torch.utils.data.random_split(
            client_data, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # 固定隨機種子
        )
    except ValueError as e:
        print(f"Error splitting data for client {client_id}: {e}")
        return None
    
    # 確保測試數據不為空
    if len(test_data) == 0:
        print(f"Warning: No test data for client {client_id}")
        return None
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def check_data_distribution():
    """檢查數據分佈的輔助函數"""
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
    # 測試資料分佈
    check_data_distribution()
