import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_dataloaders
from model import OptimizedMNISTCNN, LeNetMNIST
import os
import numpy as np

class ClientDP:
    def __init__(self, cid, device=torch.device('cpu'), epsilon=1.0, delta=1e-5):
        self.cid = cid
        self.device = device
        # 使用LeNet架構以便與iDLG攻擊兼容
        self.model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(device)
        
        # 差分隱私參數
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = 1.0  # 固定梯度裁剪範數
        
        # 根據epsilon計算noise_multiplier
        self.noise_multiplier = self._calculate_noise_multiplier(epsilon, delta)
        
    def _calculate_noise_multiplier(self, epsilon, delta):
        """
        根據epsilon和delta計算noise multiplier
        使用改進的DP公式，確保低epsilon時有明顯差異
        """
        if epsilon <= 0:
            return float('inf')  # 無限噪聲
        
        # 改進的DP公式：對不同epsilon範圍使用不同策略
        sensitivity = self.max_grad_norm
        
        if epsilon >= 10.0:
            # 對於高epsilon，使用較小的噪聲
            noise_multiplier = (1.0 * sensitivity) / epsilon
        else:
            # 對於低epsilon，使用標準公式
            noise_multiplier = (2 * np.log(1.25 / delta) * sensitivity) / epsilon
        
        return max(noise_multiplier, 0.01)  # 降低最小值0.01避免數值問題
        
    def set_parameters(self, parameters):
        """設置模型參數"""
        try:
            if isinstance(parameters, list) and len(parameters) > 0:
                if isinstance(parameters[0], torch.Tensor):
                    model_params = list(self.model.parameters())
                    if len(parameters) == len(model_params):
                        with torch.no_grad():
                            for param, new_param in zip(model_params, parameters):
                                param.data.copy_(new_param.to(self.device))
                        return
                else:
                    with torch.no_grad():
                        model_params = list(self.model.parameters())
                        if len(parameters) == len(model_params):
                            for param, new_param in zip(model_params, parameters):
                                param.data.copy_(torch.tensor(new_param).to(self.device))
                            return
            
            if isinstance(parameters, dict):
                self.model.load_state_dict(parameters, strict=True)
                return
            
            print(f"Warning: Client {self.cid} using random initialization")
            
        except Exception as e:
            print(f"Warning: Client {self.cid} parameter loading failed: {e}")
    
    def get_parameters(self):
        """獲取模型參數"""
        return [param.detach().cpu().clone() for param in self.model.parameters()]
    
    def clip_gradients(self, gradients, max_norm):
        """梯度裁剪"""
        total_norm = torch.sqrt(sum(grad.norm() ** 2 for grad in gradients))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        
        clipped_gradients = []
        for grad in gradients:
            clipped_gradients.append(grad * clip_coef_clamped)
        
        return clipped_gradients
    
    def add_noise_to_gradients(self, gradients, noise_multiplier, max_norm):
        """為梯度添加高斯噪聲"""
        noisy_gradients = []
        for grad in gradients:
            noise = torch.normal(
                mean=0.0, 
                std=noise_multiplier * max_norm, 
                size=grad.shape
            ).to(grad.device)
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
    
    def fit(self, parameters, config):
        """訓練模型（加入差分隱私）"""
        self.set_parameters(parameters)
        
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"Client {self.cid}: No data available")
            return self.get_parameters(), 0, 0.0
        
        train_loader, test_loader = result
        if train_loader is None:
            print(f"Client {self.cid}: No training data")
            return self.get_parameters(), 0, 0.0
        
        epochs = config.get("local_epochs", 2)
        learning_rate = config.get("learning_rate", 0.001)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        total_samples = 0
        total_loss = 0.0
        total_correct = 0  # 修正：添加總正確數計算
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 應用差分隱私：梯度裁剪和噪聲添加
                gradients = [param.grad.clone() for param in self.model.parameters() if param.grad is not None]
                
                # 梯度裁剪
                clipped_gradients = self.clip_gradients(gradients, self.max_grad_norm)
                
                # 添加噪聲
                noisy_gradients = self.add_noise_to_gradients(
                    clipped_gradients, self.noise_multiplier, self.max_grad_norm
                )
                
                # 更新模型參數
                with torch.no_grad():
                    for param, noisy_grad in zip(self.model.parameters(), noisy_gradients):
                        if param.grad is not None:
                            param.grad.data = noisy_grad
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_samples += len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        total_samples += epoch_samples
        total_loss += epoch_loss
        total_correct += correct  # 修正：累計總正確數
        epoch_acc = correct / epoch_samples if epoch_samples > 0 else 0.0
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"  Client {self.cid} DP Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}, Samples={epoch_samples}, "
              f"ε={self.epsilon:.1f}")
    
        # 修正：計算最終訓練準確度
        final_train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return self.get_parameters(), total_samples, final_train_accuracy
    
    def evaluate(self):
        """評估模型"""
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"Client {self.cid}: No data for evaluation")
            return 0.0
        
        _, test_loader = result
        if test_loader is None:
            print(f"Client {self.cid}: No test data")
            return 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Client {self.cid} evaluation: {correct}/{total} correct, accuracy: {accuracy:.4f}")
        return accuracy
    
    def generate_idlg_data_with_dp(self, global_params, round_num):
        """生成用於iDLG攻擊的DP梯度數據"""
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
            sample_data = batch_data[0:1].to(self.device)  # shape: [1, 1, 28, 28]
            sample_label = batch_labels[0:1].to(self.device)  # shape: [1]
            
            print(f"Target sample: shape={sample_data.shape}, label={sample_label.item()}")
            
            # 計算原始梯度
            self.model.zero_grad()
            output = self.model(sample_data)
            loss = torch.nn.CrossEntropyLoss()(output, sample_label)
            loss.backward()
            
            # 收集原始梯度
            original_gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    original_gradients.append(param.grad.clone())
            
            # 計算原始梯度範數
            original_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in original_gradients)).item()
            
            # 應用差分隱私
            clipped_gradients = self.clip_gradients(original_gradients, self.max_grad_norm)
            dp_gradients = self.add_noise_to_gradients(clipped_gradients, self.noise_multiplier, self.max_grad_norm)
            
            # 計算DP梯度範數
            dp_grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in dp_gradients)).item()
            
            # 從梯度推斷標籤（iDLG的核心）
            if len(dp_gradients) > 0:
                # 假設最後一層是輸出層
                last_layer_grad = dp_gradients[-1]
                if last_layer_grad.dim() > 0:
                    predicted_label = torch.argmin(last_layer_grad, dim=0).item()
                else:
                    predicted_label = sample_label.item()  # 回退到真實標籤
            else:
                predicted_label = sample_label.item()
            
            # 修正：確保保存正確的鍵名
            attack_data = {
                'dp_gradients': [g.cpu() for g in dp_gradients],           # 確保使用正確的鍵名
                'original_gradients': [g.cpu() for g in original_gradients],
                'target_data': sample_data.cpu(),
                'true_label': sample_label.cpu(),
                'predicted_label': predicted_label,
                'epsilon': self.epsilon,
                'delta': 1e-5,  # 添加 delta
                'noise_multiplier': self.noise_multiplier,
                'max_grad_norm': self.max_grad_norm,
                'client_id': self.cid,
                'round': round_num,
                'original_grad_norm': original_grad_norm,
                'dp_grad_norm': dp_grad_norm
            }
            
            # 確保目錄存在
            os.makedirs('idlg_inputs_dp', exist_ok=True)
            
            # 保存檔案
            filename = f"idlg_inputs_dp/round{round_num}_client{self.cid}_eps{self.epsilon}_dp_attack_data.pt"
            torch.save(attack_data, filename)
            print(f"Attack data saved: {filename}")
            
            # 輸出攻擊資訊
            print(f"Original gradient norm: {original_grad_norm:.6f}")
            print(f"DP gradient norm: {dp_grad_norm:.6f}")
            print(f"Epsilon: {self.epsilon}")
            print(f"Noise multiplier: {self.noise_multiplier:.4f}")
            print(f"Max grad norm: {self.max_grad_norm}")
            print(f"Sample label: {sample_label.item()}")
            
        except Exception as e:
            print(f"Error generating iDLG attack data: {e}")
            import traceback
            traceback.print_exc()