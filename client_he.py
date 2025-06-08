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
        """Set public key from server"""  
        self.public_key = public_key
        print(f"Client {self.cid} received public key")
        
    def set_parameters(self, parameters):
        """Set model parameters"""  
        self.model.load_state_dict(parameters)
        
    def get_parameters(self):
        """Get model parameters"""  
        return copy.deepcopy(self.model.state_dict())
    
    def encrypt_gradients(self, gradients, global_params, slice_num=1):
        """
        Encrypt gradients using Paillier encryption
        Based on FedBoosting implementation
        """  
        if self.public_key is None:
            raise ValueError("Public key not set!")
            
        encrypted_gradients = {}
        precision_factor = 1e8  # Precision factor, smaller than FedBoosting to avoid overflow
        
        print(f"Client {self.cid} encrypting gradients...")
        
        for name, param in gradients.items():
            if param is not None:
                # Calculate gradient difference
                grad_diff = (global_params[name] - param) / slice_num
                
                # Convert to numpy and flatten
                grad_flat = grad_diff.cpu().numpy().flatten()
                
                # Convert to integer and encrypt
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
        """Client local training"""  
        print(f"Client {self.cid} starting local training...")
        
        # Get data
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"Client {self.cid}: No data available")
            return None
        
        train_loader, _ = result
        
        # Save global parameters
        global_params = self.get_parameters()
        
        # Local training
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
        
        # Get trained parameters
        local_params = self.get_parameters()
        
        # Encrypt gradients
        encrypted_gradients = self.encrypt_gradients(local_params, global_params)
        
        print(f"Client {self.cid} training completed. Avg loss: {avg_loss:.4f}")
        
        return {
            'encrypted_gradients': encrypted_gradients,
            'num_samples': len(train_loader.dataset),
            'loss': avg_loss
        }
    
    def generate_idlg_data_with_he(self, global_params, round_num):
        """
        Generate HE-encrypted gradient data for iDLG attack
        Only saves encrypted gradients - no original gradients for true HE protection
        """
        # Set global parameters
        self.set_parameters(global_params)
        
        # Get data
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"Client {self.cid}: No data available for attack data generation")
            return
        
        train_loader, _ = result
        if train_loader is None:
            print(f"Client {self.cid}: No training data available")
            return
        
        # Get the first sample from first batch
        try:
            data_iter = iter(train_loader)
            batch_data, batch_labels = next(data_iter)
            
            # Take only the first sample
            target_data = batch_data[0:1].to(self.device)
            true_label = batch_labels[0:1].to(self.device)
            
            print(f"Target sample: shape={target_data.shape}, label={true_label.item()}")
            
            # Calculate original gradients
            self.model.zero_grad()
            output = self.model(target_data)
            loss = torch.nn.CrossEntropyLoss()(output, true_label)
            loss.backward()
            
            # Collect original gradients for encryption ONLY
            original_gradients_dict = {}
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_tensor = param.grad.clone()
                    original_gradients_dict[name] = grad_tensor   # Only for encryption
            
            # Encrypt gradients using current model parameters as "global_params"
            current_params = self.get_parameters()
            encrypted_gradients = self.encrypt_gradients(original_gradients_dict, current_params)
            
            # TRUE HE PROTECTION: Only save encrypted data
            attack_data = {
                # HE encrypted data - attacker can only access these
                'encrypted_gradients': encrypted_gradients,
                'precision_factor': 1e8,
                
                # Target data and label for attack evaluation
                'target_data': target_data.cpu(),
                'true_label': true_label.cpu(),
                
                # Metadata
                'client_id': self.cid,
                'round': round_num,
                'encryption_method': 'paillier_he',
                'he_protection': True  # Flag indicating true HE protection
                
                # NO 'gradients' field - original gradients are NOT saved!
            }
            
            # Ensure directory exists
            os.makedirs('idlg_inputs_he', exist_ok=True)
            
            # Save file with consistent naming pattern
            filename = f"idlg_inputs_he/round{round_num}_client{self.cid}_he_attack_data.pt"
            torch.save(attack_data, filename)
            print(f"HE attack data saved: {filename}")
            
            print(f"Sample label: {true_label.item()}")
            print(f"Encrypted gradients: {len(encrypted_gradients)} layer groups")
            print(f"Original gradients: NOT SAVED (true HE protection)")
            
        except Exception as e:
            print(f"Error generating HE iDLG attack data: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate(self):
        """Evaluate model"""  
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