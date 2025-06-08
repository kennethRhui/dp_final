import flwr as fl
from typing import Dict, Optional, Tuple, List
import torch
from model import LeNetMNIST
from client_dp import ClientDP
import numpy as np
import os
from dataset import get_dataloaders
from utils import aggregate_weighted_average, get_model_params, set_model_params

class FederatedServerDP:
    def __init__(self, num_clients=5, device=torch.device('cpu')):
        self.num_clients = num_clients
        self.device = device
        self.global_model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(device)
        self.current_round = 0
        
    def get_initial_parameters(self):
        """獲取初始全局參數"""
        return [param.detach().cpu().clone() for param in self.global_model.parameters()]
    
    def aggregate_parameters(self, client_parameters_list, client_weights):
        """聚合客戶端參數"""
        if not client_parameters_list:
            return self.get_initial_parameters()
        
        # 加權平均聚合
        aggregated_params = []
        total_weight = sum(client_weights)
        
        for param_idx in range(len(client_parameters_list[0])):
            weighted_param = torch.zeros_like(client_parameters_list[0][param_idx])
            
            for client_idx, client_params in enumerate(client_parameters_list):
                weight = client_weights[client_idx] / total_weight
                weighted_param += weight * client_params[param_idx]
            
            aggregated_params.append(weighted_param)
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params):
        """更新全局模型"""
        with torch.no_grad():
            for param, new_param in zip(self.global_model.parameters(), aggregated_params):
                param.data.copy_(new_param.to(self.device))

def run_federated_training_with_multiple_epsilon():
    """
    運行5種不同epsilon值的差分隱私聯邦學習訓練
    """
    print("Starting Multi-Epsilon Differential Privacy Federated Learning")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 重新調整epsilon值：確保有明顯的保護差異
    epsilon_values = [
        100.0,  # 極低隱私保護（幾乎無保護）
        10.0,   # 低隱私保護
        1.0,    # 中等隱私保護
        0.1,    # 高隱私保護
        0.01    # 極高隱私保護
    ]
    
    num_clients = 5
    num_rounds = 5
    delta = 1e-5  # 固定delta值
    
    print(f"Testing {len(epsilon_values)} epsilon values: {epsilon_values}")
    print(f"Delta: {delta}")
    print(f"Expected to generate {len(epsilon_values)} attack data files")
    
    for epsilon_idx, epsilon in enumerate(epsilon_values):
        print(f"\n{'='*60}")
        print(f"TESTING EPSILON = {epsilon} (Test {epsilon_idx + 1}/{len(epsilon_values)})")
        print(f"{'='*60}")
        
        # 為每個epsilon創建新的服務器和客戶端
        server = FederatedServerDP(num_clients=num_clients, device=device)
        clients = [
            ClientDP(cid=i, device=device, epsilon=epsilon, delta=delta) 
            for i in range(num_clients)
        ]
        
        # 計算預期的噪聲倍數
        expected_noise = (2 * np.log(1.25 / delta)) / epsilon if epsilon > 0 else float('inf')
        print(f"Initialized {num_clients} clients with ε={epsilon}, δ={delta}")
        print(f"Expected noise multiplier: {expected_noise:.4f}")
        
        # 獲取初始參數
        global_params = server.get_initial_parameters()
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} (ε={epsilon}) ---")
            
            # 客戶端訓練
            client_params_list = []
            client_weights = []
            client_accuracies = []
            
            for client in clients:
                # 根據epsilon調整訓練配置
                if epsilon >= 10.0:
                    # 低保護：適中訓練輪數和學習率
                    local_epochs = 3
                    learning_rate = 0.01
                elif epsilon >= 1.0:
                    # 中等保護：適中訓練參數
                    local_epochs = 5
                    learning_rate = 0.005
                else:
                    # 高保護：更多訓練輪數補償噪聲
                    local_epochs = 8
                    learning_rate = 0.001
                
                config = {
                    "local_epochs": local_epochs,
                    "learning_rate": learning_rate
                }
                
                # 客戶端訓練
                updated_params, num_samples, train_accuracy = client.fit(global_params, config)
                
                if num_samples > 0:
                    client_params_list.append(updated_params)
                    client_weights.append(num_samples)
                    
                    # 獲取客戶端評估準確度
                    eval_accuracy = client.evaluate()
                    client_accuracies.append(eval_accuracy)
                    
                    print(f"Client {client.cid}: {num_samples} samples, "
                          f"train_acc: {train_accuracy:.4f}, eval_acc: {eval_accuracy:.4f}")
                    
                    # 只為對角線模式的客戶端生成攻擊數據（節省時間）
                    # 且只在特定輪次生成（為了區分不同epsilon）
                    if client.cid == round_num and round_num == epsilon_idx:
                        client.generate_idlg_data_with_dp(global_params, round_num)
                        print(f"Generated DP attack data for ε={epsilon}, Round {round_num}, Client {client.cid}")
            
            # 參數聚合
            if client_params_list:
                aggregated_params = server.aggregate_parameters(client_params_list, client_weights)
                server.update_global_model(aggregated_params)
                global_params = aggregated_params
                
                total_samples = sum(client_weights)
                
                # 修正：正確計算加權平均準確度
                if client_accuracies:
                    weighted_accuracy = sum(w * acc for w, acc in zip(client_weights, client_accuracies))
                    avg_accuracy = weighted_accuracy / total_samples
                else:
                    avg_accuracy = 0.0
                
                print(f"Round {round_num + 1} completed: {len(client_params_list)}/{num_clients} clients")
                print(f"Total samples: {total_samples}")
                print(f"Weighted average accuracy: {avg_accuracy:.4f}")
                print(f"Individual accuracies: {[f'{acc:.4f}' for acc in client_accuracies]}")
            
            server.current_round = round_num + 1
        
        print(f"Completed training with ε={epsilon}")
        
        # 最終整體評估
        if clients:
            final_accuracies = [client.evaluate() for client in clients]
            final_avg_accuracy = sum(final_accuracies) / len(final_accuracies)
            print(f"Final average accuracy for ε={epsilon}: {final_avg_accuracy:.4f}")
            print(f"Final individual accuracies: {[f'{acc:.4f}' for acc in final_accuracies]}")
    
    print(f"\nMulti-epsilon federated learning completed!")
    print(f"DP iDLG attack data saved in 'idlg_inputs_dp/' directory")
    
    # 列出生成的攻擊數據文件
    if os.path.exists("idlg_inputs_dp"):
        attack_files = [f for f in os.listdir("idlg_inputs_dp") if f.endswith('.pt')]
        print(f"Generated {len(attack_files)} DP attack data files:")
        for file in sorted(attack_files):
            print(f"   - {file}")
    
    return epsilon_values

if __name__ == "__main__":
    epsilon_values = run_federated_training_with_multiple_epsilon()
    print(f"\nReady for DP iDLG attacks on {len(epsilon_values)} different epsilon values!")
    print(f"Epsilon values tested: {epsilon_values}")
