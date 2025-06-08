import flwr as fl
from typing import Dict, Optional, Tuple, List
import torch
from model import LeNetMNIST
from client import Client
import numpy as np
import os
from dataset import get_dataloaders
from utils import aggregate_weighted_average, get_model_params, set_model_params

class FederatedServer:
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

def run_federated_training_with_idlg():
    """
    運行5輪聯邦學習訓練並生成25個iDLG攻擊數據
    """
    print("🚀 Starting 5-Round Federated Learning with iDLG Attack Data Generation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化服務器和客戶端
    num_clients = 5
    num_rounds = 5  # 修改為5輪
    server = FederatedServer(num_clients=num_clients, device=device)
    clients = [Client(cid=i, device=device) for i in range(num_clients)]
    
    print(f"📊 Server initialized with {num_clients} clients")
    print(f"🔄 Planning {num_rounds} federated learning rounds")
    print(f"🎯 Expected to generate {num_clients * num_rounds} attack data files")
    
    # 獲取初始參數
    global_params = server.get_initial_parameters()
    
    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"FEDERATED LEARNING ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*50}")
        
        # 客戶端訓練
        client_params_list = []
        client_weights = []
        
        for client in clients:
            print(f"\n--- Client {client.cid} Training ---")
            
            # 訓練配置
            config = {
                "local_epochs": 2,
                "learning_rate": 0.001
            }
            
            # 客戶端訓練
            updated_params, num_samples, accuracy = client.fit(global_params, config)
            
            if num_samples > 0:
                client_params_list.append(updated_params)
                client_weights.append(num_samples)
                print(f"✅ Client {client.cid}: {num_samples} samples, accuracy: {accuracy:.4f}")
                
                # 每輪每個客戶端都生成攻擊數據（總共25個）
                client.generate_idlg_data(global_params, round_num)
            else:
                print(f"❌ Client {client.cid}: No training data")
        
        # 參數聚合
        if client_params_list:
            aggregated_params = server.aggregate_parameters(client_params_list, client_weights)
            server.update_global_model(aggregated_params)
            global_params = aggregated_params
            
            total_samples = sum(client_weights)
            avg_weight = total_samples / len(client_weights) if client_weights else 0
            print(f"\n📊 Round {round_num + 1} Summary:")
            print(f"   Participating clients: {len(client_params_list)}/{num_clients}")
            print(f"   Total samples: {total_samples}")
            print(f"   Average samples per client: {avg_weight:.1f}")
        else:
            print(f"❌ Round {round_num + 1}: No valid client updates")
        
        server.current_round = round_num + 1
    
    print(f"\n✅ Federated learning completed!")
    print(f"📁 iDLG attack data saved in 'idlg_inputs/' directory")
    
    # 列出生成的攻擊數據文件
    if os.path.exists("idlg_inputs"):
        attack_files = [f for f in os.listdir("idlg_inputs") if f.endswith('.pt')]
        print(f"📋 Generated {len(attack_files)} attack data files:")
        for file in sorted(attack_files):
            print(f"   - {file}")
    
    return global_params

if __name__ == "__main__":
    final_params = run_federated_training_with_idlg()
    print("\n🎯 Ready for iDLG attacks on 25 data files!")
