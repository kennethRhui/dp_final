import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def get_model_params(model):
    """
    獲取模型參數
    """
    return [param.data.clone() for param in model.parameters()]

def set_model_params(model, params):
    """
    設置模型參數
    """
    for param, new_param in zip(model.parameters(), params):
        param.data = new_param.data.clone()

def aggregate_weighted_average(client_params_list, client_weights=None):
    """
    使用加權平均聚合客戶端參數
    
    Args:
        client_params_list: 客戶端參數列表的列表
        client_weights: 客戶端權重列表（如果為None，則使用等權重）
    
    Returns:
        list: 聚合後的參數列表
    """
    if not client_params_list:
        return []
    
    num_clients = len(client_params_list)
    
    # 如果沒有提供權重，使用等權重
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        # 歸一化權重
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    # 獲取參數數量
    num_params = len(client_params_list[0])
    aggregated_params = []
    
    for param_idx in range(num_params):
        # 收集所有客戶端的該參數
        param_tensors = [client_params[param_idx] for client_params in client_params_list]
        
        # 加權平均
        weighted_param = torch.zeros_like(param_tensors[0])
        for param_tensor, weight in zip(param_tensors, client_weights):
            weighted_param += param_tensor * weight
        
        aggregated_params.append(weighted_param)
    
    return aggregated_params

def save_gradients_list(gradients, filepath):
    """
    保存梯度列表到文件
    
    Args:
        gradients: 梯度列表
        filepath: 保存路径
    """
    try:
        # 確保目錄存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 將梯度轉換為可保存的格式
        gradients_to_save = []
        for grad in gradients:
            if isinstance(grad, torch.Tensor):
                gradients_to_save.append(grad.detach().cpu())
            else:
                gradients_to_save.append(grad)
        
        torch.save(gradients_to_save, filepath)
        print(f"Gradients saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving gradients to {filepath}: {e}")

def load_gradients_list(filepath):
    """
    從文件載入梯度列表
    
    Args:
        filepath: 文件路径
        
    Returns:
        list: 梯度列表
    """
    try:
        if not os.path.exists(filepath):
            print(f"Gradient file not found: {filepath}")
            return None
        
        gradients = torch.load(filepath, map_location='cpu')
        print(f"Gradients loaded from {filepath}")
        return gradients
        
    except Exception as e:
        print(f"Error loading gradients from {filepath}: {e}")
        return None

def load_gradient(filepath):
    """
    從文件載入單個梯度（為了向後兼容）
    
    Args:
        filepath: 文件路径
        
    Returns:
        torch.Tensor or list: 梯度數據
    """
    try:
        if not os.path.exists(filepath):
            print(f"Gradient file not found: {filepath}")
            return None
        
        gradient = torch.load(filepath, map_location='cpu')
        print(f"Gradient loaded from {filepath}")
        return gradient
        
    except Exception as e:
        print(f"Error loading gradient from {filepath}: {e}")
        return None

def save_gradient(gradient, filepath):
    """
    保存單個梯度到文件
    
    Args:
        gradient: 梯度數據
        filepath: 保存路径
    """
    try:
        # 確保目錄存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 將梯度轉換為可保存的格式
        if isinstance(gradient, torch.Tensor):
            gradient_to_save = gradient.detach().cpu()
        else:
            gradient_to_save = gradient
        
        torch.save(gradient_to_save, filepath)
        print(f"Gradient saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving gradient to {filepath}: {e}")

def calculate_psnr(img1, img2):
    """
    計算兩個圖像之間的 PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 第一個圖像張量 (C, H, W) 或 (H, W)
        img2: 第二個圖像張量 (C, H, W) 或 (H, W)
    
    Returns:
        float: PSNR 值 (單位: dB)
    """
    # 確保輸入是 torch.Tensor 並在 CPU 上
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)
    
    # 轉移到 CPU
    img1 = img1.cpu()
    img2 = img2.cpu()
    
    # 確保形狀匹配
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        return 0.0
    
    # 計算 MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # 避免除零錯誤
    if mse == 0:
        return float('inf')
    
    # 計算 PSNR
    max_pixel_value = 1.0  # 假設圖像已經歸一化到 [0, 1]
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    計算兩個圖像之間的 SSIM (Structural Similarity Index)
    
    Args:
        img1: 第一個圖像張量 (C, H, W) 或 (H, W)
        img2: 第二個圖像張量 (C, H, W) 或 (H, W)
    
    Returns:
        float: SSIM 值 (範圍: [-1, 1])
    """
    # 確保張量在 CPU 上並轉換為 numpy 數組
    try:
        if isinstance(img1, torch.Tensor):
            img1_np = img1.detach().cpu().numpy()
        else:
            img1_np = np.array(img1)
        
        if isinstance(img2, torch.Tensor):
            img2_np = img2.detach().cpu().numpy()
        else:
            img2_np = np.array(img2)
    except Exception as e:
        print(f"Error converting tensors to numpy: {e}")
        return 0.0
    
    # 確保形狀匹配
    if img1_np.shape != img2_np.shape:
        print(f"Warning: Image shapes don't match: {img1_np.shape} vs {img2_np.shape}")
        return 0.0
    
    # 確保數據類型正確
    img1_np = img1_np.astype(np.float64)
    img2_np = img2_np.astype(np.float64)
    
    # 確保數值範圍在 [0, 1]
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # 處理不同的圖像維度
    if len(img1_np.shape) == 3:  # (C, H, W)
        if img1_np.shape[0] == 1:  # 單通道
            img1_np = img1_np.squeeze(0)
            img2_np = img2_np.squeeze(0)
        else:  # 多通道
            # 對於多通道圖像，計算每個通道的 SSIM 然後取平均
            ssim_values = []
            for c in range(img1_np.shape[0]):
                try:
                    channel_ssim = compute_ssim_2d(img1_np[c], img2_np[c])
                    ssim_values.append(channel_ssim)
                except Exception as e:
                    print(f"Warning: SSIM calculation failed for channel {c}: {e}")
                    ssim_values.append(0.0)
            return np.mean(ssim_values)
    
    # 單通道或已經是 2D 圖像
    return compute_ssim_2d(img1_np, img2_np)

def compute_ssim_2d(img1, img2):
    """
    計算 2D 圖像的 SSIM
    """
    try:
        # 檢查圖像大小
        min_dim = min(img1.shape)
        if min_dim < 7:
            # 對於小圖像，使用較小的窗口
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            win_size = max(3, win_size)  # 至少使用 3x3 窗口
        else:
            win_size = 7
        
        # 計算 SSIM
        ssim_val = ssim(
            img1, img2, 
            data_range=1.0,  # 數據範圍 [0, 1]
            win_size=win_size
        )
        
        return ssim_val
        
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        # 如果 scikit-image 失敗，使用簡單的相關係數作為替代
        return simple_ssim(img1, img2)

def simple_ssim(img1, img2):
    """
    簡單的 SSIM 替代實現（基於相關係數）
    """
    try:
        # 計算均值
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # 計算方差和協方差
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM 的簡化版本
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
        
    except Exception as e:
        print(f"Simple SSIM calculation failed: {e}")
        return 0.0

def aggregate_gradients(client_gradients_list, aggregation_method='avg'):
    """
    聚合來自多個客戶端的梯度
    
    Args:
        client_gradients_list: 客戶端梯度列表的列表
        aggregation_method: 聚合方法 ('avg', 'sum', 'weighted_avg')
    
    Returns:
        list: 聚合後的梯度列表
    """
    if not client_gradients_list:
        return []
    
    # 獲取參數數量
    num_params = len(client_gradients_list[0])
    num_clients = len(client_gradients_list)
    
    aggregated_grads = []
    
    for param_idx in range(num_params):
        # 收集所有客戶端的該參數梯度
        param_grads = [client_grads[param_idx] for client_grads in client_gradients_list]
        
        if aggregation_method == 'avg':
            # 平均聚合
            aggregated_grad = torch.stack(param_grads).mean(dim=0)
        elif aggregation_method == 'sum':
            # 求和聚合
            aggregated_grad = torch.stack(param_grads).sum(dim=0)
        else:  # 默認使用平均
            aggregated_grad = torch.stack(param_grads).mean(dim=0)
        
        aggregated_grads.append(aggregated_grad)
    
    return aggregated_grads

def apply_differential_privacy(gradients, noise_multiplier=1.0, l2_norm_clip=1.0):
    """
    對梯度應用差分隱私噪聲
    
    Args:
        gradients: 梯度列表
        noise_multiplier: 噪聲倍數
        l2_norm_clip: L2 範數裁剪閾值
    
    Returns:
        list: 添加噪聲後的梯度列表
    """
    dp_gradients = []
    
    for grad in gradients:
        # L2 範數裁剪
        grad_norm = torch.norm(grad)
        if grad_norm > l2_norm_clip:
            grad = grad * (l2_norm_clip / grad_norm)
        
        # 添加高斯噪聲
        noise_std = noise_multiplier * l2_norm_clip
        noise = torch.normal(0, noise_std, size=grad.shape)
        
        dp_grad = grad + noise
        dp_gradients.append(dp_grad)
    
    return dp_gradients

def normalize_image(img):
    """
    將圖像歸一化到 [0, 1] 範圍
    
    Args:
        img: 輸入圖像張量
    
    Returns:
        torch.Tensor: 歸一化後的圖像
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu()
    
    img_min = img.min()
    img_max = img.max()
    
    if img_max == img_min:
        return torch.zeros_like(img)
    
    return (img - img_min) / (img_max - img_min)

def tensor_to_numpy(tensor):
    """
    將 PyTorch 張量轉換為 NumPy 數組
    
    Args:
        tensor: PyTorch 張量
    
    Returns:
        numpy.ndarray: NumPy 數組
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def numpy_to_tensor(array, device=None):
    """
    將 NumPy 數組轉換為 PyTorch 張量
    
    Args:
        array: NumPy 數組
        device: 目標設備
    
    Returns:
        torch.Tensor: PyTorch 張量
    """
    tensor = torch.from_numpy(array).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def clip_image(img, min_val=0.0, max_val=1.0):
    """
    將圖像值裁剪到指定範圍
    
    Args:
        img: 輸入圖像
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        裁剪後的圖像
    """
    if isinstance(img, torch.Tensor):
        return torch.clamp(img, min_val, max_val)
    else:
        return np.clip(img, min_val, max_val)