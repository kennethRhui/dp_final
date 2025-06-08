import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def get_model_params(model):
    """
    Get model parameters
    """
    return [param.data.clone() for param in model.parameters()]

def set_model_params(model, params):
    """
    Set model parameters
    """
    for param, new_param in zip(model.parameters(), params):
        param.data = new_param.data.clone()

def aggregate_weighted_average(client_params_list, client_weights=None):
    """
    Aggregate client parameters using weighted average
    
    Args:
        client_params_list: List of client parameter lists
        client_weights: List of client weights (if None, use equal weights)
    
    Returns:
        list: Aggregated parameter list
    """
    if not client_params_list:
        return []
    
    num_clients = len(client_params_list)
    
    # If no weights provided, use equal weights
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    # Get number of parameters
    num_params = len(client_params_list[0])
    aggregated_params = []
    
    for param_idx in range(num_params):
        # Collect this parameter from all clients
        param_tensors = [client_params[param_idx] for client_params in client_params_list]
        
        # Weighted average
        weighted_param = torch.zeros_like(param_tensors[0])
        for param_tensor, weight in zip(param_tensors, client_weights):
            weighted_param += param_tensor * weight
        
        aggregated_params.append(weighted_param)
    
    return aggregated_params

def save_gradients_list(gradients, filepath):
    """
    Save gradient list to file
    
    Args:
        gradients: Gradient list
        filepath: Save path
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert gradients to saveable format
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
    Load gradient list from file
    
    Args:
        filepath: File path
        
    Returns:
        list: Gradient list
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
    Load single gradient from file (for backward compatibility)
    
    Args:
        filepath: File path
        
    Returns:
        torch.Tensor or list: Gradient data
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
    Save single gradient to file
    
    Args:
        gradient: Gradient data
        filepath: Save path
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert gradient to saveable format
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
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images
    
    Args:
        img1: First image tensor (C, H, W) or (H, W)
        img2: Second image tensor (C, H, W) or (H, W)
    
    Returns:
        float: PSNR value (unit: dB)
    """
    # Ensure inputs are torch.Tensor and on CPU
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)
    
    # Move to CPU
    img1 = img1.cpu()
    img2 = img2.cpu()
    
    # Ensure shapes match
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        return 0.0
    
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel_value = 1.0  # Assume images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images
    
    Args:
        img1: First image tensor (C, H, W) or (H, W)
        img2: Second image tensor (C, H, W) or (H, W)
    
    Returns:
        float: SSIM value (range: [-1, 1])
    """
    # Ensure tensors are on CPU and convert to numpy arrays
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
    
    # Ensure shapes match
    if img1_np.shape != img2_np.shape:
        print(f"Warning: Image shapes don't match: {img1_np.shape} vs {img2_np.shape}")
        return 0.0
    
    # Ensure correct data type
    img1_np = img1_np.astype(np.float64)
    img2_np = img2_np.astype(np.float64)
    
    # Ensure values are in [0, 1] range
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Handle different image dimensions
    if len(img1_np.shape) == 3:  # (C, H, W)
        if img1_np.shape[0] == 1:  # Single channel
            img1_np = img1_np.squeeze(0)
            img2_np = img2_np.squeeze(0)
        else:  # Multi-channel
            # For multi-channel images, calculate SSIM for each channel and take average
            ssim_values = []
            for c in range(img1_np.shape[0]):
                try:
                    channel_ssim = compute_ssim_2d(img1_np[c], img2_np[c])
                    ssim_values.append(channel_ssim)
                except Exception as e:
                    print(f"Warning: SSIM calculation failed for channel {c}: {e}")
                    ssim_values.append(0.0)
            return np.mean(ssim_values)
    
    # Single channel or already 2D image
    return compute_ssim_2d(img1_np, img2_np)

def compute_ssim_2d(img1, img2):
    """
    Calculate SSIM for 2D images
    """
    try:
        # Check image size
        min_dim = min(img1.shape)
        if min_dim < 7:
            # For small images, use smaller window
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            win_size = max(3, win_size)  # Use at least 3x3 window
        else:
            win_size = 7
        
        # Calculate SSIM
        ssim_val = ssim(
            img1, img2, 
            data_range=1.0,  # Data range [0, 1]
            win_size=win_size
        )
        
        return ssim_val
        
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        # If scikit-image fails, use simple correlation coefficient as alternative
        return simple_ssim(img1, img2)

def simple_ssim(img1, img2):
    """
    Simple SSIM alternative implementation (based on correlation coefficient)
    """
    try:
        # Calculate means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # Calculate variance and covariance
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        # Simplified SSIM
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
    Aggregate gradients from multiple clients
    
    Args:
        client_gradients_list: List of client gradient lists
        aggregation_method: Aggregation method ('avg', 'sum', 'weighted_avg')
    
    Returns:
        list: Aggregated gradient list
    """
    if not client_gradients_list:
        return []
    
    # Get number of parameters
    num_params = len(client_gradients_list[0])
    num_clients = len(client_gradients_list)
    
    aggregated_grads = []
    
    for param_idx in range(num_params):
        # Collect this parameter gradient from all clients
        param_grads = [client_grads[param_idx] for client_grads in client_gradients_list]
        
        if aggregation_method == 'avg':
            # Average aggregation
            aggregated_grad = torch.stack(param_grads).mean(dim=0)
        elif aggregation_method == 'sum':
            # Sum aggregation
            aggregated_grad = torch.stack(param_grads).sum(dim=0)
        else:  # Default to average
            aggregated_grad = torch.stack(param_grads).mean(dim=0)
        
        aggregated_grads.append(aggregated_grad)
    
    return aggregated_grads

def apply_differential_privacy(gradients, noise_multiplier=1.0, l2_norm_clip=1.0):
    """
    Apply differential privacy noise to gradients
    
    Args:
        gradients: Gradient list
        noise_multiplier: Noise multiplier
        l2_norm_clip: L2 norm clipping threshold
    
    Returns:
        list: Gradient list with added noise
    """
    dp_gradients = []
    
    for grad in gradients:
        # L2 norm clipping
        grad_norm = torch.norm(grad)
        if grad_norm > l2_norm_clip:
            grad = grad * (l2_norm_clip / grad_norm)
        
        # Add Gaussian noise
        noise_std = noise_multiplier * l2_norm_clip
        noise = torch.normal(0, noise_std, size=grad.shape)
        
        dp_grad = grad + noise
        dp_gradients.append(dp_grad)
    
    return dp_gradients

def normalize_image(img):
    """
    Normalize image to [0, 1] range
    
    Args:
        img: Input image tensor
    
    Returns:
        torch.Tensor: Normalized image
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
    Convert PyTorch tensor to NumPy array
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        numpy.ndarray: NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def numpy_to_tensor(array, device=None):
    """
    Convert NumPy array to PyTorch tensor
    
    Args:
        array: NumPy array
        device: Target device
    
    Returns:
        torch.Tensor: PyTorch tensor
    """
    tensor = torch.from_numpy(array).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def clip_image(img, min_val=0.0, max_val=1.0):
    """
    Clip image values to specified range
    
    Args:
        img: Input image
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clipped image
    """
    if isinstance(img, torch.Tensor):
        return torch.clamp(img, min_val, max_val)
    else:
        return np.clip(img, min_val, max_val)