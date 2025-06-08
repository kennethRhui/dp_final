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
        
        # Generate Paillier keypair
        print("Generating Paillier keypair...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=512)
        print("Paillier keypair generated successfully")
        
        # Initialize clients
        self.clients = []
        for i in range(num_clients):
            client = ClientHE(i, device)
            client.set_public_key(self.public_key)
            self.clients.append(client)
    
    def get_parameters(self):
        """Get global model parameters"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_parameters(self, parameters):
        """Set global model parameters"""
        self.model.load_state_dict(parameters)
    
    def decrypt_gradients(self, encrypted_gradients, global_params):
        """
        Decrypt encrypted gradients
        Based on FedBoosting implementation
        """
        decrypted_gradients = {}
        
        for name, encrypted_data in encrypted_gradients.items():
            encrypted_list = encrypted_data['encrypted_data']
            shape = encrypted_data['shape']
            precision_factor = encrypted_data['precision_factor']
            
            # Decrypt and convert back to float
            decrypted_list = []
            for encrypted_value in encrypted_list:
                decrypted_int = self.private_key.decrypt(encrypted_value)
                decrypted_float = decrypted_int / precision_factor
                decrypted_list.append(decrypted_float)
            
            # Reshape and rebuild original weights
            decrypted_array = np.array(decrypted_list).reshape(shape)
            decrypted_tensor = torch.from_numpy(decrypted_array).float()
            
            # Rebuild weights: global_params - gradient_diff
            decrypted_gradients[name] = global_params[name] - decrypted_tensor.to(self.device)
        
        return decrypted_gradients
    
    def aggregate_parameters(self, client_updates):
        """Aggregate client parameters"""
        if not client_updates:
            return self.get_parameters()
        
        # Get current global parameters
        global_params = self.get_parameters()
        
        # Decrypt all client gradients
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
        
        # Weighted average aggregation
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
        """Execute one round of federated learning"""
        print(f"\n{'='*50}")
        print(f"Round {round_num + 1}/{5}") 
        print(f"{'='*50}")
        
        # Distribute global model parameters to all clients
        global_params = self.get_parameters()
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Client local training
        client_updates = []
        client_accuracies = []
        
        for client in self.clients:
            print(f"\n--- Client {client.cid} Training ---")  
            
            update = client.train(epochs)
            client_updates.append(update)
            
            # Generate HE attack data(each client generates data each round)
            if update and update['num_samples'] > 0:
                # Evaluate client
                eval_accuracy = client.evaluate()
                client_accuracies.append(eval_accuracy)
                print(f"Client {client.cid}: {update['num_samples']} samples, "
                      f"loss: {update['loss']:.4f}, eval_acc: {eval_accuracy:.4f}")
                
                # Generate HE attack data for each client each round
                client.generate_idlg_data_with_he(global_params, round_num)
                print(f"Generated HE attack data for Round {round_num}, Client {client.cid}")
            else:
                print(f"Client {client.cid}: No training data")
        
        # Aggregate parameters
        aggregated_params = self.aggregate_parameters(client_updates)
        self.set_parameters(aggregated_params)
        
        # Calculate average metrics
        valid_updates = [update for update in client_updates if update is not None]
        avg_loss = np.mean([update['loss'] for update in valid_updates]) if valid_updates else 0.0
        avg_accuracy = np.mean(client_accuracies) if client_accuracies else 0.0
        
        print(f"\nRound {round_num + 1} Summary:")
        print(f"   Participating clients: {len(valid_updates)}/{self.num_clients}")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Average accuracy: {avg_accuracy:.4f}")
        
        return avg_loss
    
    def evaluate(self):
        """Evaluate global model"""
        print("\nEvaluating global model...")
        accuracies = []
        
        for client in self.clients:
            client.set_parameters(self.get_parameters())
            accuracy = client.evaluate()
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        print(f"Global model average accuracy: {avg_accuracy:.4f}")
        return avg_accuracy

def run_federated_learning_with_he():
    """
    Run 5-round federated learning with Homomorphic Encryption protection
    and generate 25 HE attack data files
    """
    print("Starting 5-Round Federated Learning with Homomorphic Encryption Protection")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create server and clients
    num_clients = 5
    num_rounds = 5
    server = ServerHE(device, num_clients=num_clients)
    
    print(f"Server initialized with {num_clients} clients")
    print(f"Planning {num_rounds} federated learning rounds")
    print(f"Expected to generate {num_clients * num_rounds} HE attack data files")
    
    print(f"\nTraining for {num_rounds} rounds with HE protection...")
    
    # Execute federated learning
    for round_num in range(num_rounds):
        loss = server.train_round(round_num, epochs=1)  
        
        # Evaluate every round
        if round_num % 1 == 0:
            accuracy = server.evaluate()
    
    print(f"\nHomomorphic Encryption Protected Federated Learning completed!")
    print(f"HE attack data saved in 'idlg_inputs_he/' directory")
    
    # List generated HE attack data files
    import os
    if os.path.exists("idlg_inputs_he"):
        attack_files = [f for f in os.listdir("idlg_inputs_he") if f.endswith('.pt')]
        print(f"Generated {len(attack_files)} HE attack data files:")
        for file in sorted(attack_files):
            print(f"   - {file}")
    
    print("\nReady for iDLG attacks on HE-protected data!")
    print("Next step: Run 'python idlg_attack_he.py' to perform attacks")

if __name__ == "__main__":
    run_federated_learning_with_he()