import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_dataloaders
from model import OptimizedMNISTCNN, LeNetMNIST
import os
import numpy as np

class Client:
    def __init__(self, cid, device=torch.device('cpu')):
        self.cid = cid
        self.device = device
        # Use LeNet architecture for compatibility with iDLG attacks
        self.model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(device)
        
    def set_parameters(self, parameters):
        """Set model parameters"""
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
        """Get model parameters"""
        return [param.detach().cpu().clone() for param in self.model.parameters()]
    
    def fit(self, parameters, config):
        """Train model"""
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_samples += len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
            total_samples += epoch_samples
            total_loss += epoch_loss
            epoch_acc = correct / epoch_samples if epoch_samples > 0 else 0.0
            avg_loss = epoch_loss / len(train_loader)
            
            print(f"  Client {self.cid} Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}, Samples={epoch_samples}")
        
        final_accuracy = self.evaluate()
        return self.get_parameters(), total_samples, final_accuracy
    
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
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def generate_idlg_data(self, global_params, round_num):
        """
        Generate data for iDLG attacks
        Simulate real federated learning training process
        """
        print(f"Generating iDLG attack data for Client {self.cid}, Round {round_num}")
        
        # Set global parameters
        self.set_parameters(global_params)
        
        # Get training data
        result = get_dataloaders(self.cid)
        if result is None:
            print(f"No data available for Client {self.cid}")
            return
        
        train_loader, _ = result
        if train_loader is None:
            print(f"No training data for Client {self.cid}")
            return
        
        # Get single batch for attack
        for data_batch, target_batch in train_loader:
            # Take only the first sample
            gt_data = data_batch[0:1].to(self.device)
            gt_label = target_batch[0:1].to(self.device)
            
            print(f"Target sample: shape={gt_data.shape}, label={gt_label.item()}")
            
            # Ensure model is in training mode (consistent with real federated learning)
            self.model.train()
            
            # Calculate real gradients
            criterion = torch.nn.CrossEntropyLoss()
            
            # Forward pass
            output = self.model(gt_data)
            loss = criterion(output, gt_label)
            
            # Calculate gradients
            original_gradients = torch.autograd.grad(loss, self.model.parameters())
            original_gradients = [grad.detach().clone().cpu() for grad in original_gradients]
            
            # Save attack data
            os.makedirs("idlg_inputs", exist_ok=True)
            
            attack_data = {
                'gt_data': gt_data.detach().cpu(),
                'gt_label': gt_label.detach().cpu(),
                'gradients': original_gradients,
                'model_state': self.model.state_dict(),
                'loss_value': loss.item(),
                'round': round_num,
                'client_id': self.cid
            }
            
            # Save data
            save_path = f"idlg_inputs/round{round_num}_client{self.cid}_attack_data.pt"
            torch.save(attack_data, save_path)
            
            # Verify gradients
            total_norm = sum(grad.norm().item() for grad in original_gradients)
            print(f"Attack data saved: {save_path}")
            print(f"Gradient norm: {total_norm:.6f}")
            print(f"Sample label: {gt_label.item()}")
            
            break  # Only process first batch