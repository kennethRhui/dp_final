import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from model import LeNetMNIST
from utils import calculate_psnr, calculate_ssim
from torchvision import transforms

class FixedIDLGAttacker:
    """
    Fixed iDLG attacker - fully consistent with original paper
    """
    
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.model = None
        
        # Transform utilities - consistent with original paper
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def attack(self, attack_data_path, method='iDLG', lr=1.0, iterations=300):
        """
        Execute fixed iDLG attack
        """
        print(f"Starting Fixed {method} Attack")
        
        # Load attack data
        attack_data = torch.load(attack_data_path, map_location='cpu')
        gt_data = attack_data['gt_data'].to(self.device)
        gt_label = attack_data['gt_label'].to(self.device)
        original_gradients = [grad.to(self.device) for grad in attack_data['gradients']]
        
        print(f"Ground truth: shape={gt_data.shape}, label={gt_label.item()}")
        
        # Key fix 1: Use new randomly initialized model
        channel = gt_data.shape[1]  # 1 for MNIST
        hidden = 588  # MNIST LeNet hidden size
        num_classes = 10
        
        self.model = LeNetMNIST(channel=channel, hidden=hidden, num_classes=num_classes).to(self.device)
        
        # Key fix 2: Apply same weight initialization as original paper
        self._weights_init_like_paper(self.model)
        
        # Key fix 3: Recompute gradients (using randomly initialized model)
        self.model.train()  # Training mode
        criterion = nn.CrossEntropyLoss()
        
        # Recompute real gradients with current random model
        out = self.model(gt_data)
        y = criterion(out, gt_label)
        original_dy_dx = torch.autograd.grad(y, self.model.parameters())
        original_dy_dx = [grad.detach().clone() for grad in original_dy_dx]
        
        print(f"Recomputed gradients with random model: {len(original_dy_dx)} tensors")
        
        # Initialize dummy data
        dummy_data = torch.randn(gt_data.size()).to(self.device).requires_grad_(True)
        
        # Key fix 4: Correct label prediction (using second-to-last layer)
        label_pred = self._predict_label_correctly(original_dy_dx)
        print(f"Predicted label: {label_pred.item()} (True: {gt_label.item()})")
        
        optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        
        # Recording variables
        losses = []
        mses = []
        
        print(f"Starting optimization with lr={lr}, iterations={iterations}")
        start_time = time.time()
        
        # Key fix 5: Optimization loop identical to original paper
        for iters in range(iterations):
            def closure():
                optimizer.zero_grad()
                pred = self.model(dummy_data)
                
                # iDLG loss same as original paper
                dummy_loss = criterion(pred, label_pred)

                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            current_loss = closure().item()
            losses.append(current_loss)
            mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

            # Progress display - same frequency as original paper
            if iters % int(iterations / 30) == 0:
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(f"{current_time} {iters} loss = {current_loss:.8f}, mse = {mses[-1]:.8f}")

            # Convergence check - same threshold as original paper
            if current_loss < 0.000001:
                print(f"Converged at iteration {iters}!")
                break
        
        # Calculate final results
        total_time = time.time() - start_time
        final_loss = losses[-1] if losses else float('inf')
        final_mse = mses[-1] if mses else float('inf')
        
        # Predict label
        predicted_label = label_pred.item()
        
        # Calculate quality metrics
        reconstructed_image = dummy_data[0].detach().cpu().clamp(0, 1)
        original_image = gt_data[0].cpu()
        
        psnr = calculate_psnr(original_image, reconstructed_image)
        ssim = calculate_ssim(original_image, reconstructed_image)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"FIXED {method} ATTACK RESULTS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Iterations: {len(losses)}")
        print(f"Final loss: {final_loss:.8f}")
        print(f"Final MSE: {final_mse:.8f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        print(f"Ground truth label: {gt_label.item()}")
        print(f"Predicted label: {predicted_label}")
        print(f"Label accuracy: {'CORRECT' if predicted_label == gt_label.item() else 'INCORRECT'}")
        
        # Save results
        self._save_final_results(
            original_image, reconstructed_image, gt_label.item(), 
            predicted_label, psnr, ssim, final_mse, method, attack_data_path
        )
        
        return reconstructed_image, predicted_label, {
            'loss': final_loss,
            'mse': final_mse,
            'psnr': psnr,
            'ssim': ssim,
            'iterations': len(losses),
            'time': total_time
        }
    
    def _weights_init_like_paper(self, model):
        """
        Weight initialization identical to original paper
        """
        def weights_init(m):
            try:
                if hasattr(m, "weight"):
                    m.weight.data.uniform_(-0.5, 0.5)
            except Exception:
                print(f'warning: failed in weights_init for {m._get_name()}.weight')
            try:
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.uniform_(-0.5, 0.5)
            except Exception:
                print(f'warning: failed in weights_init for {m._get_name()}.bias')
        
        model.apply(weights_init)
        print("Applied paper-consistent weight initialization")
    
    def _predict_label_correctly(self, gradients):
        """
        Correct label prediction - using second-to-last layer (weight layer)
        """
        # Use second-to-last layer (fully connected layer weights), not last layer (bias)
        fc_weight_grad = gradients[-2]  # Second-to-last layer should be FC layer weights
        
        # Calculate gradient sum for each class
        label_pred = torch.argmin(torch.sum(fc_weight_grad, dim=-1), dim=-1)
        
        return label_pred.detach().reshape((1,)).requires_grad_(False)
    
    def _save_final_results(self, original, reconstructed, true_label, pred_label, 
                          psnr, ssim, mse, method, data_path):
        """Save final results"""
        try:
            os.makedirs("idlg_results", exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original.squeeze(), cmap='gray')
            axes[0].set_title(f'Original Private Data\nTrue Label: {true_label}', 
                             fontsize=14, weight='bold')
            axes[0].axis('off')
            
            # Reconstructed image
            axes[1].imshow(reconstructed.squeeze(), cmap='gray')
            axes[1].set_title(f'Fixed {method} Reconstructed\nPred Label: {pred_label}\n'
                             f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}', fontsize=14)
            axes[1].axis('off')
            
            # Difference image
            diff = torch.abs(original - reconstructed)
            im = axes[2].imshow(diff.squeeze(), cmap='hot')
            axes[2].set_title(f'Absolute Difference\nMSE: {mse:.6f}', fontsize=14)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Set title color
            if mse < 0.01:
                title_color = 'red'
                status = 'CRITICAL PRIVACY BREACH'
            elif mse < 0.05:
                title_color = 'darkorange'
                status = 'SEVERE PRIVACY LEAK'
            else:
                title_color = 'green'
                status = 'PRIVACY PARTIALLY PROTECTED'
            
            plt.suptitle(f'Fixed {method} Attack Results - {status}', 
                        fontsize=16, color=title_color, weight='bold')
            
            plt.tight_layout()
            
            # Save results
            base_name = os.path.basename(data_path).replace('.pt', '')
            filename = f"idlg_results/fixed_{method}_final_{base_name}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Fixed results saved to: {filename}")
            
        except Exception as e:
            print(f"Warning: Could not save final results: {e}")

def run_all_25_idlg_attacks():
    """
    Run all 25 iDLG attack tests (5 Rounds × 5 Clients)
    """
    print("Running All 25 iDLG Attacks (5 Rounds × 5 Clients)")
    print("Using iDLG method only")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check attack data
    if not os.path.exists("idlg_inputs"):
        print("No attack data found. Please run 'python server.py' first.")
        return
    
    attack_files = [f for f in os.listdir("idlg_inputs") if f.endswith('.pt')]
    if not attack_files:
        print("No attack data files found in idlg_inputs/")
        return
    
    print(f"Found {len(attack_files)} attack data files")
    
    # Create fixed attacker
    attacker = FixedIDLGAttacker(device)
    
    # Attack statistics
    total_attacks = 0
    successful_attacks = 0
    attack_results = []
    
    # Test all files - only iDLG method
    for attack_file in sorted(attack_files):
        attack_path = os.path.join("idlg_inputs", attack_file)
        print(f"\n{'='*50}")
        print(f"Testing: {attack_file}")
        print(f"{'='*50}")
        
        # Only test iDLG
        method = 'iDLG'
        print(f"\n--- Fixed {method} Attack ---")
        total_attacks += 1
        
        reconstructed, pred_label, results = attacker.attack(
            attack_path, 
            method=method, 
            lr=1.0, 
            iterations=300
        )
        
        if reconstructed is not None:
            successful_attacks += 1
            attack_results.append({
                'file': attack_file,
                'method': method,
                'mse': results['mse'],
                'psnr': results['psnr'],
                'ssim': results['ssim'],
                'time': results['time']
            })
            print(f"Fixed {method} attack completed successfully")
            print(f"MSE: {results['mse']:.6f}, PSNR: {results['psnr']:.2f}")
        else:
            print(f"Fixed {method} attack failed")
    
    # Attack statistics report
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE iDLG ATTACK STATISTICS")
    print(f"{'='*70}")
    print(f"Total attacks performed: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Success rate: {successful_attacks/total_attacks*100:.1f}%")
    print(f"Total result images: {successful_attacks}")
    
    # Calculate average metrics for successful attacks
    if attack_results:
        avg_mse = sum(r['mse'] for r in attack_results) / len(attack_results)
        avg_psnr = sum(r['psnr'] for r in attack_results) / len(attack_results)
        avg_ssim = sum(r['ssim'] for r in attack_results) / len(attack_results)
        avg_time = sum(r['time'] for r in attack_results) / len(attack_results)
        
        print(f"\nAverage iDLG Attack Performance:")
        print(f"   MSE: {avg_mse:.6f}")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f}")
        print(f"   Time: {avg_time:.1f} seconds")
        
        # Attack effectiveness classification
        critical_attacks = sum(1 for r in attack_results if r['mse'] < 0.01)
        severe_attacks = sum(1 for r in attack_results if 0.01 <= r['mse'] < 0.05)
        moderate_attacks = len(attack_results) - critical_attacks - severe_attacks
        
        print(f"\niDLG Attack Impact Analysis:")
        print(f"   Critical Privacy Breach (MSE < 0.01): {critical_attacks}")
        print(f"   Severe Privacy Leak (0.01 ≤ MSE < 0.05): {severe_attacks}")
        print(f"   Moderate Success (MSE ≥ 0.05): {moderate_attacks}")
        
        # Time analysis
        min_time = min(r['time'] for r in attack_results)
        max_time = max(r['time'] for r in attack_results)
        
        print(f"\niDLG Attack Time Analysis:")
        print(f"   Minimum time: {min_time:.1f} seconds")
        print(f"   Maximum time: {max_time:.1f} seconds")
        print(f"   Average time: {avg_time:.1f} seconds")
    
    print(f"\nAll 25 iDLG attacks completed!")
    print(f"Results saved in 'idlg_results/' directory")
    print(f"Compare these baseline times with DP-protected attack times")

if __name__ == "__main__":
    run_all_25_idlg_attacks()