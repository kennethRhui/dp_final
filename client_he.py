import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import phe as paillier
from dataset import get_dataloaders
from model import LeNetMNIST
import os

class ClientHE:
    def __init__(self, cid, device):
        self.cid = cid
        self.device = device
        self.model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(device)
        self.public_key = None
        
    def set_public_key(self, public_key):
        """設置來自服務器的公鑰"""
        self.public_key = public_key
        print(f"Client {self.cid} received public key")
        
    def set_parameters(self, parameters):
        """設置模型參數"""
        self.model.load_state_dict(parameters)
        
    def get_parameters(self):
        """獲取模型參數"""
        return copy.deepcopy(self.model.state_dict())
    
    def encrypt_gradients(self, gradients, global_params, slice_num=1):
        """
        使用 Paillier 加密梯度
        基於 FedBoosting 的實現方式
        """
        if self.public_key is None:
            raise ValueError("Public key not set!")
            
        encrypted_gradients = {}
        precision_factor = 1e8  # 精度係數，比 FedBoosting 小一點避免溢出
        
        print(f"Client {self.cid} encrypting gradients...")
        
        for name, param in gradients.items():
            if param is not None:
                # 計算梯度差
                grad_diff = (global_params[name] - param) / slice_num
                
                # 轉換為numpy並平坦化
                grad_flat = grad_diff.cpu().numpy().flatten()
                
                # 轉換為整數並加密
                encrypted_list = []
                for value in grad_flat:
                    int_value = int(value * precision_factor)
                    encrypted_value = self.public_key.encrypt(int_value)
                    encrypted_list.append(encrypted_value)
                
                encrypted_gradients[name] = {
                    'encrypted_data': encrypted_list,
                    'shape': grad_diff.shape,
                    'precision_factor': precision_factor
                }
        
        print(f"Client {self.cid} gradient encryption completed")
        return encrypted_gradients
    
    def train(self, epochs=1):
        """客戶端本地訓練"""
        print(f"Client {self.cid} starting local training...")
        
        # 獲取數據
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"Client {self.cid}: No data available")
            return None
        
        train_loader, _ = result
        
        # 保存全局參數
        global_params = self.get_parameters()
        
        # 本地訓練
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 獲取訓練後的參數
        local_params = self.get_parameters()
        
        # 加密梯度
        encrypted_gradients = self.encrypt_gradients(local_params, global_params)
        
        print(f"Client {self.cid} training completed. Avg loss: {avg_loss:.4f}")
        
        return {
            'encrypted_gradients': encrypted_gradients,
            'num_samples': len(train_loader.dataset),
            'loss': avg_loss
        }
    
    def generate_idlg_data_with_he(self, global_params, round_num):
        """生成用於iDLG攻擊的HE加密梯度數據"""
        # 設置全局參數
        self.set_parameters(global_params)
        
        # 獲取一個樣本
        result = get_dataloaders(self.cid)
        if result is None:
            return
        
        train_loader, _ = result
        if train_loader is None:
            return
        
        # 獲取第一個batch的第一個樣本
        try:
            data_iter = iter(train_loader)
            batch_data, batch_labels = next(data_iter)
            
            # 只取第一個樣本
            sample_data = batch_data[0:1].to(self.device)
            sample_label = batch_labels[0:1].to(self.device)
            
            print(f"Target sample: shape={sample_data.shape}, label={sample_label.item()}")
            
            # 計算原始梯度
            self.model.zero_grad()
            output = self.model(sample_data)
            loss = torch.nn.CrossEntropyLoss()(output, sample_label)
            loss.backward()
            
            # 收集原始梯度
            original_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    original_gradients[name] = param.grad.clone()
            
            # 加密梯度
            encrypted_gradients = self.encrypt_gradients(original_gradients, global_params)
            
            # 保存攻擊數據
            attack_data = {
                'encrypted_gradients': encrypted_gradients,
                'target_data': sample_data.cpu(),
                'true_label': sample_label.cpu(),
                'client_id': self.cid,
                'round': round_num,
                'precision_factor': 1e8
            }
            
            # 確保目錄存在
            os.makedirs('idlg_inputs_he', exist_ok=True)
            
            # 保存檔案
            filename = f"idlg_inputs_he/round{round_num}_client{self.cid}_he_attack_data.pt"
            torch.save(attack_data, filename)
            print(f"HE attack data saved: {filename}")
            
            # 輸出攻擊資訊
            print(f"Sample label: {sample_label.item()}")
            print(f"Encrypted gradients keys: {list(encrypted_gradients.keys())}")
            
        except Exception as e:
            print(f"Error generating HE iDLG attack data: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate(self):
        """評估模型"""
        result = get_dataloaders(self.cid)
        if result is None:
            return 0.0
        
        _, test_loader = result
        if test_loader is None:
            return 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Client {self.cid} evaluation: {correct}/{total} correct, accuracy: {accuracy:.4f}")
        return accuracy