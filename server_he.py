import torch
import numpy as np
import phe as paillier
from model import LeNetMNIST
import copy
from client_he import ClientHE

class ServerHE:
    def __init__(self, device, num_clients=5):
        self.device = device
        self.model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(device)
        self.num_clients = num_clients
        
        # 生成 Paillier 密鑰對
        print("Generating Paillier keypair...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=128)
        print("Paillier keypair generated successfully")
        
        # 初始化客戶端
        self.clients = []
        for i in range(num_clients):
            client = ClientHE(i, device)
            client.set_public_key(self.public_key)
            self.clients.append(client)
    
    def get_parameters(self):
        """獲取全局模型參數"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_parameters(self, parameters):
        """設置全局模型參數"""
        self.model.load_state_dict(parameters)
    
    def decrypt_gradients(self, encrypted_gradients, global_params):
        """
        解密加密的梯度
        基於 FedBoosting 的實現方式
        """
        decrypted_gradients = {}
        
        for name, encrypted_data in encrypted_gradients.items():
            encrypted_list = encrypted_data['encrypted_data']
            shape = encrypted_data['shape']
            precision_factor = encrypted_data['precision_factor']
            
            # 解密並轉換回浮點數
            decrypted_list = []
            for encrypted_value in encrypted_list:
                decrypted_int = self.private_key.decrypt(encrypted_value)
                decrypted_float = decrypted_int / precision_factor
                decrypted_list.append(decrypted_float)
            
            # 重塑形狀並重建原始權重
            decrypted_array = np.array(decrypted_list).reshape(shape)
            decrypted_tensor = torch.from_numpy(decrypted_array).float()
            
            # 重建權重：global_params - gradient_diff
            decrypted_gradients[name] = global_params[name] - decrypted_tensor.to(self.device)
        
        return decrypted_gradients
    
    def aggregate_parameters(self, client_updates):
        """聚合客戶端參數"""
        if not client_updates:
            return self.get_parameters()
        
        # 獲取當前全局參數
        global_params = self.get_parameters()
        
        # 解密所有客戶端的梯度
        all_decrypted_params = []
        total_samples = 0
        
        for update in client_updates:
            if update and 'encrypted_gradients' in update:
                decrypted_params = self.decrypt_gradients(
                    update['encrypted_gradients'], 
                    global_params
                )
                all_decrypted_params.append({
                    'params': decrypted_params,
                    'num_samples': update['num_samples']
                })
                total_samples += update['num_samples']
        
        if not all_decrypted_params:
            return global_params
        
        # 加權平均聚合
        aggregated_params = {}
        for name in global_params.keys():
            weighted_sum = torch.zeros_like(global_params[name])
            
            for client_data in all_decrypted_params:
                if name in client_data['params']:
                    weight = client_data['num_samples'] / total_samples
                    weighted_sum += client_data['params'][name] * weight
            
            aggregated_params[name] = weighted_sum
        
        return aggregated_params
    
    def train_round(self, round_num, epochs=1):
        """執行一輪聯邦學習"""
        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print(f"{'='*50}")
        
        # 分發全局模型參數給所有客戶端
        global_params = self.get_parameters()
        for client in self.clients:
            client.set_parameters(global_params)
        
        # 生成 HE 攻擊數據（對角線模式）
        if round_num < len(self.clients):
            target_client = self.clients[round_num]
            target_client.generate_idlg_data_with_he(global_params, round_num)
        
        # 客戶端本地訓練
        client_updates = []
        for client in self.clients:
            update = client.train(epochs)
            client_updates.append(update)
        
        # 聚合參數
        aggregated_params = self.aggregate_parameters(client_updates)
        self.set_parameters(aggregated_params)
        
        # 計算平均損失
        avg_loss = np.mean([
            update['loss'] for update in client_updates 
            if update is not None
        ])
        
        print(f"Round {round_num} completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self):
        """評估全局模型"""
        print("\nEvaluating global model...")
        accuracies = []
        
        for client in self.clients:
            client.set_parameters(self.get_parameters())
            accuracy = client.evaluate()
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        print(f"Global model average accuracy: {avg_accuracy:.4f}")
        return avg_accuracy

def run_federated_learning_he():
    """運行同態加密保護的聯邦學習"""
    print("Starting Homomorphic Encryption Protected Federated Learning")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 創建服務器
    server = ServerHE(device, num_clients=5)
    
    # 訓練輪數
    num_rounds = 5
    
    print(f"\nTraining for {num_rounds} rounds with HE protection...")
    
    # 執行聯邦學習
    for round_num in range(num_rounds):
        loss = server.train_round(round_num, epochs=1)
        
        # 每輪評估
        if round_num % 1 == 0:
            accuracy = server.evaluate()
    
    print(f"\nHomomorphic Encryption Protected Federated Learning completed!")
    print(f"HE attack data saved in 'idlg_inputs_he/' directory")

if __name__ == "__main__":
    run_federated_learning_he()