import torch
import tenseal as ts
from model import CNN
from client_he import ClientHE
from utils import get_model_params, set_model_params
import os

# HE 配置參數
POLY_MODULUS_DEGREE = 16384  # 增加以避免警告
GLOBAL_SCALE = 2**20

# FL 配置參數
NUM_ROUNDS_HE = 5
NUM_CLIENTS_HE = 5
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.01
GLOBAL_LEARNING_RATE = 1.0

def initialize_he_context():
    """初始化同態加密上下文"""
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MODULUS_DEGREE
        )
        context.global_scale = GLOBAL_SCALE
        context.generate_galois_keys()
        context.generate_relin_keys()
        
        print(f"HE Context initialized successfully:")
        print(f"  Poly modulus degree: {POLY_MODULUS_DEGREE}")
        print(f"  Global scale: {GLOBAL_SCALE}")
        
        return context
    except Exception as e:
        print(f"Error initializing HE context: {e}")
        raise
def aggregate_encrypted_gradients(all_clients_encrypted_grads, context):
    """聚合加密梯度"""
    if not all_clients_encrypted_grads:
        return []
    
    print("Aggregating encrypted gradients...")
    num_clients = len(all_clients_encrypted_grads)
    
    # 確定參數數量（從第一個客戶端）
    num_params = len(all_clients_encrypted_grads[0])
    aggregated_grads = []
    
    for param_idx in range(num_params):
        print(f"Aggregating parameter {param_idx}...")
        
        # 收集所有客戶端的該參數梯度
        client_encrypted_grads = []
        original_shape = None
        
        for client_grads in all_clients_encrypted_grads:
            if param_idx < len(client_grads):
                grad_info = client_grads[param_idx]
                if grad_info['encrypted_data'] is not None:
                    # 重要：將加密梯度的上下文設置為伺服器上下文
                    encrypted_grad = grad_info['encrypted_data']
                    encrypted_grad.link_context(context)
                    client_encrypted_grads.append(encrypted_grad)
                    if original_shape is None:
                        original_shape = grad_info['original_shape']
        
        if not client_encrypted_grads:
            print(f"No valid encrypted gradients for parameter {param_idx}")
            aggregated_grads.append({
                'param_idx': param_idx,
                'original_shape': original_shape,
                'aggregated_encrypted': None
            })
            continue
        
        # 同態聚合：加法
        try:
            aggregated_encrypted = client_encrypted_grads[0].copy()
            for i in range(1, len(client_encrypted_grads)):
                aggregated_encrypted += client_encrypted_grads[i]
            
            # 平均：乘以 1/n
            aggregated_encrypted *= (1.0 / num_clients)
            
            aggregated_grads.append({
                'param_idx': param_idx,
                'original_shape': original_shape,
                'aggregated_encrypted': aggregated_encrypted
            })
            
        except Exception as e:
            print(f"Error aggregating parameter {param_idx}: {e}")
            aggregated_grads.append({
                'param_idx': param_idx,
                'original_shape': original_shape,
                'aggregated_encrypted': None
            })
    
    return aggregated_grads

def decrypt_and_reshape_gradients(aggregated_encrypted_grads, context):
    """解密並重塑梯度"""
    print("Decrypting aggregated gradients...")
    decrypted_grads = []
    
    for grad_info in aggregated_encrypted_grads:
        param_idx = grad_info['param_idx']
        original_shape = grad_info['original_shape']
        encrypted_grad = grad_info['aggregated_encrypted']
        
        if encrypted_grad is None:
            print(f"Parameter {param_idx}: encrypted gradient is None, using zeros")
            decrypted_tensor = torch.zeros(original_shape, dtype=torch.float32)
        else:
            try:
                # 確保加密梯度連接到正確的上下文
                encrypted_grad.link_context(context)
                
                # 解密
                decrypted_list = encrypted_grad.decrypt()
                
                # 轉換為張量並重塑
                expected_numel = torch.Size(original_shape).numel()
                
                # 處理長度不匹配
                if len(decrypted_list) > expected_numel:
                    decrypted_list = decrypted_list[:expected_numel]
                elif len(decrypted_list) < expected_numel:
                    decrypted_list.extend([0.0] * (expected_numel - len(decrypted_list)))
                
                decrypted_tensor = torch.tensor(decrypted_list, dtype=torch.float32)
                decrypted_tensor = decrypted_tensor.reshape(original_shape)
                
                print(f"Parameter {param_idx}: Successfully decrypted, "
                      f"shape {original_shape}, "
                      f"stats: min={decrypted_tensor.min().item():.6e}, "
                      f"max={decrypted_tensor.max().item():.6e}, "
                      f"norm={torch.norm(decrypted_tensor).item():.6e}")
                
            except Exception as e:
                print(f"Error decrypting parameter {param_idx}: {e}")
                decrypted_tensor = torch.zeros(original_shape, dtype=torch.float32)
        
        decrypted_grads.append(decrypted_tensor)
    
    return decrypted_grads

def save_gradients_for_attack(decrypted_grads, round_num, target_client_id=0):
    """儲存解密後的梯度用於攻擊驗證"""
    os.makedirs("inputs_he", exist_ok=True)
    
    gradient_path = os.path.join(
        "inputs_he", 
        f"round{round_num}_client{target_client_id}_aggregated_decrypted_gradient_he.pt"
    )
    
    torch.save(decrypted_grads, gradient_path)
    print(f"Saved decrypted aggregated gradients to {gradient_path}")

def update_global_model(global_model, decrypted_grads, learning_rate):
    """更新全局模型"""
    print("Updating global model with decrypted gradients...")
    
    current_params = get_model_params(global_model)
    updated_params = []
    
    total_gradient_norm = 0.0
    for i, (param, grad) in enumerate(zip(current_params, decrypted_grads)):
        grad_norm = torch.norm(grad).item()
        total_gradient_norm += grad_norm
        
        updated_param = param - learning_rate * grad.to(param.device)
        updated_params.append(updated_param)
        print(f"Updated parameter {i}, gradient norm: {grad_norm:.6e}")
    
    set_model_params(global_model, updated_params)
    print(f"Global model updated successfully. Total gradient norm: {total_gradient_norm:.6e}")

def evaluate_global_model(global_model):
    """評估全局模型在測試集上的性能"""
    try:
        from dataset import get_dataloaders
        
        result = get_dataloaders(0, batch_size=64)
        if result is None:
            print("Warning: get_dataloaders returned None, skipping evaluation")
            return 0.0
        
        train_loader, test_loader = result
        if test_loader is None:
            print("Warning: test_loader is None, using train_loader for evaluation")
            test_loader = train_loader
        
        if test_loader is None:
            print("Warning: Both loaders are None, skipping evaluation")
            return 0.0
        
        global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = global_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Global model test accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        print("Skipping evaluation, returning default accuracy of 0.1")
        return 0.1

if __name__ == "__main__":
    print("=" * 60)
    print("Starting HE-Protected Federated Learning")
    print("=" * 60)
    
    # 初始化 HE 上下文（伺服器端持有私鑰）
    server_context = initialize_he_context()
    
    # 序列化公開上下文（不包含私鑰）給客戶端
    public_context_bytes = server_context.serialize(
        save_public_key=True,
        save_secret_key=False,  # 客戶端不需要私鑰
        save_galois_keys=True,
        save_relin_keys=True
    )
    
    # 初始化全局模型
    global_model = CNN()
    print(f"Global model initialized with {sum(p.numel() for p in global_model.parameters())} parameters")
    
    # 評估初始模型性能
    initial_accuracy = evaluate_global_model(global_model)
    print(f"Initial global model accuracy: {initial_accuracy:.4f}")
    
    # 初始化客戶端
    clients = [
        ClientHE(
            cid=i,
            tenseal_context_bytes=public_context_bytes,
            device=torch.device("cpu")
        )
        for i in range(NUM_CLIENTS_HE)
    ]
    
    # 聯邦學習主循環
    for round_num in range(NUM_ROUNDS_HE):
        print(f"\n{'='*20} Round {round_num + 1}/{NUM_ROUNDS_HE} {'='*20}")
        
        all_clients_encrypted_grads = []
        client_data_sizes = []
        client_accuracies = []
        
        # 獲取當前全局模型參數
        global_params = get_model_params(global_model)
        
        # 客戶端訓練
        for client in clients:
            print(f"\n--- Training Client {client.cid} ---")
            
            config = {
                'round': round_num,
                'local_epochs': LOCAL_EPOCHS,
                'learning_rate': LEARNING_RATE
            }
            
            encrypted_grads, data_size, accuracy = client.fit_he(global_params, config)
            
            if encrypted_grads:
                all_clients_encrypted_grads.append(encrypted_grads)
                client_data_sizes.append(data_size)
                client_accuracies.append(accuracy)
                print(f"Client {client.cid}: Sent encrypted gradients, "
                      f"data size: {data_size}, accuracy: {accuracy:.4f}")
            else:
                print(f"Warning: Client {client.cid} returned no encrypted gradients")
        
        if not all_clients_encrypted_grads:
            print("No encrypted gradients received. Skipping this round.")
            continue
        
        # 聚合加密梯度（使用伺服器上下文）
        aggregated_encrypted_grads = aggregate_encrypted_gradients(
            all_clients_encrypted_grads, server_context
        )
        
        # 解密並重塑梯度（使用伺服器上下文）
        decrypted_grads = decrypt_and_reshape_gradients(
            aggregated_encrypted_grads, server_context
        )
        
        # 儲存梯度用於攻擊驗證
        save_gradients_for_attack(decrypted_grads, round_num)
        
        # 更新全局模型
        update_global_model(global_model, decrypted_grads, GLOBAL_LEARNING_RATE)
        
        # 評估更新後的全局模型
        global_accuracy = evaluate_global_model(global_model)
        
        # 打印統計信息
        if client_accuracies:
            avg_client_accuracy = sum(client_accuracies) / len(client_accuracies)
            print(f"\nRound {round_num + 1} Summary:")
            print(f"  Average client accuracy: {avg_client_accuracy:.4f}")
            print(f"  Global model accuracy: {global_accuracy:.4f}")
            print(f"  Participating clients: {len(client_accuracies)}")
    
    # 儲存最終模型
    torch.save(global_model.state_dict(), "global_model_he_final.pth")
    
    # 最終評估
    final_accuracy = evaluate_global_model(global_model)
    print(f"\n{'='*60}")
    print("HE-Protected Federated Learning Completed!")
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Accuracy improvement: {final_accuracy - initial_accuracy:.4f}")
    print(f"Final global model saved to global_model_he_final.pth")
    print("=" * 60)