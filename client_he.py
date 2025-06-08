import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts
from model import CNN 
from dataset import get_dataloaders
from utils import set_model_params, get_model_params
import os

class ClientHE:
    def __init__(self, cid, tenseal_context_bytes=None, device=torch.device("cpu")):
        self.cid = cid
        self.device = device
        self.model = CNN().to(self.device) 
        
        # 嘗試獲取數據載入器，添加錯誤處理
        try:
            result = get_dataloaders(self.cid, batch_size=16)
            if result is None:
                print(f"Warning: Client {self.cid} - get_dataloaders returned None")
                self.train_loader = None
                self.test_loader = None
            else:
                self.train_loader, self.test_loader = result
        except Exception as e:
            print(f"Error getting dataloaders for client {self.cid}: {e}")
            self.train_loader = None
            self.test_loader = None

        if tenseal_context_bytes:
            self.context = ts.context_from(tenseal_context_bytes)
            if not self.context:
                raise ValueError(f"Client {self.cid}: Failed to create TenSEAL context from bytes.")
        else:
            raise ValueError(f"Client {self.cid}: TenSEAL context bytes are required.")

        # 儲存最後一批數據用於攻擊驗證
        self.last_x_for_idlg = None
        self.last_y_for_idlg = None

    def fit_he(self, global_model_params, config):
        """使用 HE 進行聯邦學習的訓練"""
        if self.train_loader is None:
            print(f"Client {self.cid}: No training data available, skipping training")
            return [], 0, 0.0
        
        set_model_params(self.model, global_model_params)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=config.get('learning_rate', 0.01))
        criterion = nn.CrossEntropyLoss()
        local_epochs = config.get('local_epochs', 1)
        current_round = config.get('round', 0)
        
        print(f"Client {self.cid}: Starting HE local training for {local_epochs} epochs...")

        # 訓練多個 epochs
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 儲存最後一批數據用於 iDLG 攻擊驗證
                self.last_x_for_idlg = data.detach().cpu()
                self.last_y_for_idlg = target.detach().cpu()

        # 計算最終梯度（基於最後一批數據）
        if self.last_x_for_idlg is not None and self.last_y_for_idlg is not None:
            # 重新計算最後一批的梯度（用於攻擊）
            self.model.eval()
            optimizer.zero_grad()
            
            # 只取最後一批的第一個樣本進行梯度計算（模擬 iDLG 攻擊場景）
            single_data = self.last_x_for_idlg[0:1].to(self.device)  # 取第一個樣本
            single_target = self.last_y_for_idlg[0:1].to(self.device)
            
            output = self.model(single_data)
            loss = criterion(output, single_target)
            loss.backward()

        # 收集並加密梯度
        encrypted_gradients = self._encrypt_gradients()
        
        # 儲存原始數據用於攻擊驗證
        self._save_data_for_attack(current_round)
        
        # 評估客戶端效能
        client_acc = self._evaluate()
        
        # 計算數據集大小
        data_size = len(self.train_loader.dataset) if self.train_loader else 0
        
        return encrypted_gradients, data_size, client_acc

    def _encrypt_gradients(self):
        """加密梯度並返回加密後的梯度列表"""
        encrypted_grads = []
        
        print(f"Client {self.cid}: Encrypting gradients...")
        
        for param_idx, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                # 獲取梯度並展平
                grad_flat = param.grad.detach().cpu().flatten().tolist()
                
                # 打印梯度統計信息
                grad_tensor = param.grad.detach().cpu().flatten()
                print(f"Client {self.cid}: Param {param_idx} (shape {param.grad.shape}), "
                      f"Grad stats: min={grad_tensor.min().item():.6e}, "
                      f"max={grad_tensor.max().item():.6e}, "
                      f"mean={grad_tensor.mean().item():.6e}, "
                      f"std={grad_tensor.std().item():.6e}")
                
                try:
                    # 使用 CKKS 加密梯度
                    encrypted_grad = ts.ckks_vector(self.context, grad_flat)
                    encrypted_grads.append({
                        'param_idx': param_idx,
                        'original_shape': param.grad.shape,
                        'encrypted_data': encrypted_grad
                    })
                except Exception as e:
                    print(f"Client {self.cid}: Error encrypting param {param_idx}: {e}")
                    encrypted_grads.append({
                        'param_idx': param_idx,
                        'original_shape': param.grad.shape,
                        'encrypted_data': None
                    })
            else:
                print(f"Client {self.cid}: Param {param_idx} has no gradient.")
        
        print(f"Client {self.cid}: Encrypted {len(encrypted_grads)} gradient parameters.")
        return encrypted_grads

    def _save_data_for_attack(self, current_round):
        """儲存數據用於 iDLG 攻擊驗證"""
        if self.last_x_for_idlg is not None and self.last_y_for_idlg is not None:
            os.makedirs("inputs_he", exist_ok=True)
            
            # 儲存第一個樣本
            image_to_save = self.last_x_for_idlg[0].cpu()  # 形狀: (C, H, W)
            label_to_save = self.last_y_for_idlg[0].cpu()
            
            data_path = os.path.join("inputs_he", f"round{current_round}_client{self.cid}_data_he.pt")
            torch.save({'image': image_to_save, 'label': label_to_save}, data_path)
            print(f"Client {self.cid}: Saved data for iDLG attack to {data_path}")

    def _evaluate(self):
        """評估客戶端模型效能"""
        try:
            # 檢查是否有可用的測試數據
            eval_loader = self.test_loader
            if eval_loader is None:
                print(f"Client {self.cid}: No test loader, using train loader for evaluation")
                eval_loader = self.train_loader
            
            if eval_loader is None:
                print(f"Client {self.cid}: No data loader available for evaluation")
                return 0.0
            
            self.model.eval()
            correct, total = 0, 0
            
            with torch.no_grad():
                for data, target in eval_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            print(f"Client {self.cid}: Evaluation accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            print(f"Client {self.cid}: Error during evaluation: {e}")
            return 0.0