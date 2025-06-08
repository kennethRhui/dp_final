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
        """Get initial global parameters"""
        return [param.detach().cpu().clone() for param in self.global_model.parameters()]
    
    def aggregate_parameters(self, client_parameters_list, client_weights):
        """Aggregate client parameters"""
        if not client_parameters_list:
            return self.get_initial_parameters()
        
        # Weighted average aggregation
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
        """Update global model"""
        with torch.no_grad():
            for param, new_param in zip(self.global_model.parameters(), aggregated_params):
                param.data.copy_(new_param.to(self.device))

def run_federated_training_with_idlg():
    """
    Run 5-round federated learning training and generate 25 iDLG attack data files
    """
    print("Starting 5-Round Federated Learning with iDLG Attack Data Generation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize server and clients
    num_clients = 5
    num_rounds = 5  # Modified to 5 rounds
    server = FederatedServer(num_clients=num_clients, device=device)
    clients = [Client(cid=i, device=device) for i in range(num_clients)]
    
    print(f"Server initialized with {num_clients} clients")
    print(f"Planning {num_rounds} federated learning rounds")
    print(f"Expected to generate {num_clients * num_rounds} iDLG attack data files")
    
    # Get initial parameters
    global_params = server.get_initial_parameters()
    
    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"FEDERATED LEARNING ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*50}")
        
        # Client training
        client_params_list = []
        client_weights = []
        
        for client in clients:
            print(f"\n--- Client {client.cid} Training ---")
            
            # Training configuration
            config = {
                "local_epochs": 2,
                "learning_rate": 0.001
            }
            
            # Client training
            updated_params, num_samples, accuracy = client.fit(global_params, config)
            
            if num_samples > 0:
                client_params_list.append(updated_params)
                client_weights.append(num_samples)
                print(f"Client {client.cid}: {num_samples} samples, accuracy: {accuracy:.4f}")
                
                # Each client generates iDLG attack data every round (total 25 files)
                client.generate_idlg_data(global_params, round_num)
            else:
                print(f"Client {client.cid}: No training data")
        
        # Parameter aggregation
        if client_params_list:
            aggregated_params = server.aggregate_parameters(client_params_list, client_weights)
            server.update_global_model(aggregated_params)
            global_params = aggregated_params
            
            total_samples = sum(client_weights)
            avg_weight = total_samples / len(client_weights) if client_weights else 0
            print(f"\nRound {round_num + 1} Summary:")
            print(f"   Participating clients: {len(client_params_list)}/{num_clients}")
            print(f"   Total samples: {total_samples}")
            print(f"   Average samples per client: {avg_weight:.1f}")
        else:
            print(f"Round {round_num + 1}: No valid client updates")
        
        server.current_round = round_num + 1
    
    print(f"\nFederated learning completed!")
    print(f"iDLG attack data saved in 'idlg_inputs/' directory")
    
    # List generated iDLG attack data files
    if os.path.exists("idlg_inputs"):
        attack_files = [f for f in os.listdir("idlg_inputs") if f.endswith('.pt')]
        print(f"Generated {len(attack_files)} iDLG attack data files:")
        for file in sorted(attack_files):
            print(f"   - {file}")
    
    return global_params

if __name__ == "__main__":
    final_params = run_federated_training_with_idlg()
    print("\nReady for iDLG attacks on 25 data files!")
    print("Next step: Run 'python idlg_attack.py' to perform attacks")