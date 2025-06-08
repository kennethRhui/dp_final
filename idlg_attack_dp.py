import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim
from model import LeNetMNIST

class FixedIDLGAttackerDP:
    def __init__(self, device):
        self.device = device
        
    def attack(self, attack_data_path, method='iDLG', lr=1.0, iterations=300):
        """
        Standard iDLG attack implementation - attack DP-protected gradients
        """
        try:
            # Load attack data
            attack_data = torch.load(attack_data_path, map_location=self.device)
            
            # Extract necessary information
            dp_gradients = [g.to(self.device) for g in attack_data['dp_gradients']]
            target_data = attack_data['target_data'].to(self.device)
            true_label = attack_data['true_label'].to(self.device)
            epsilon = attack_data['epsilon']
            noise_multiplier = attack_data['noise_multiplier']
            
            print(f"Loaded attack data: ε={epsilon}, noise_multiplier={noise_multiplier:.4f}")
            print(f"Target shape: {target_data.shape}, True label: {true_label.item()}")
            
            # Create model and initialize - same as original idlg_attack.py
            channel = target_data.shape[1]  # 1 for MNIST
            hidden = 588  # MNIST LeNet hidden size
            num_classes = 10
            
            model = LeNetMNIST(channel=channel, hidden=hidden, num_classes=num_classes).to(self.device)
            
            # Apply same weight initialization as original paper
            self._weights_init_like_paper(model)
            
            # Initialize dummy data - same as original
            dummy_data = torch.randn(target_data.size()).to(self.device).requires_grad_(True)
            
            # Key fix 4: Correct label prediction (using DP gradients directly)
            label_pred = self._predict_label_correctly(dp_gradients)
            print(f"Predicted label: {label_pred.item()} (True: {true_label.item()})")
            
            # Same optimizer as original
            optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
            
            # Recording variables
            losses = []
            mses = []
            
            print(f"Starting optimization with lr={lr}, iterations={iterations}")
            start_time = time.time()
            
            # Key fix 5: Optimization loop attacking DP gradients
            for iters in range(iterations):
                def closure():
                    optimizer.zero_grad()
                    pred = model(dummy_data)
                    
                    # iDLG loss same as original paper
                    dummy_loss = nn.CrossEntropyLoss()(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, dp_gradients):  # 攻擊 DP 梯度！
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - target_data) ** 2).item())

                # Progress display - same frequency as original paper
                if iters % int(iterations / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(f"{current_time} {iters} loss = {current_loss:.8f}, mse = {mses[-1]:.8f}")

                # Convergence check - same threshold as original paper
                if current_loss < 0.000001:
                    print(f"Converged at iteration {iters}!")
                    break
            
            attack_time = time.time() - start_time
            
            # Calculate final results
            final_loss = losses[-1] if losses else float('inf')
            final_mse = mses[-1] if mses else float('inf')
            predicted_label = label_pred.item()
            
            # Calculate quality metrics
            reconstructed_image = dummy_data[0].detach().cpu().clamp(0, 1)
            original_image = target_data[0].cpu()
            
            # Calculate PSNR and SSIM
            mse_metric = torch.mean((reconstructed_image - original_image) ** 2).item()
            psnr = 10 * np.log10(1.0 / (mse_metric + 1e-10))
            
            target_np = original_image.squeeze().numpy()
            reconstructed_np = reconstructed_image.squeeze().numpy()
            ssim_score = ssim(target_np, reconstructed_np, data_range=1.0)
            
            # Print results - same format as original
            print(f"\n{'='*60}")
            print(f"FIXED {method} ATTACK RESULTS")
            print(f"{'='*60}")
            print(f"Total time: {attack_time:.2f} seconds")
            print(f"Iterations: {len(losses)}")
            print(f"Final loss: {final_loss:.8f}")
            print(f"Final MSE: {final_mse:.8f}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim_score:.4f}")
            print(f"Ground truth label: {true_label.item()}")
            print(f"Predicted label: {predicted_label}")
            print(f"Label accuracy: {'CORRECT' if predicted_label == true_label.item() else 'INCORRECT'}")
            
            # Save results
            self._save_final_results(
                original_image, reconstructed_image, true_label.item(), 
                predicted_label, psnr, ssim_score, final_mse, method, attack_data_path
            )
            
            results = {
                'loss': final_loss,
                'mse': final_mse,
                'psnr': psnr,
                'ssim': ssim_score,
                'time': attack_time,
                'epsilon': epsilon,
                'delta': attack_data.get('delta', 1e-5),
                'noise_multiplier': noise_multiplier,
                'iterations': len(losses),
                'final_loss': final_loss
            }
            
            return reconstructed_image, predicted_label, results
            
        except Exception as e:
            print(f"Error in {method} attack: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
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
            os.makedirs("idlg_results_dp", exist_ok=True)
            
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
            filename = f"idlg_results_dp/fixed_{method}_final_{base_name}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Fixed results saved to: {filename}")
            
        except Exception as e:
            print(f"Warning: Could not save final results: {e}")

def run_epsilon_specific_idlg_attacks():
    """
    Run iDLG attacks on 5 specific clients per epsilon (round0 client0, round1 client1, etc.)
    Only using iDLG method to save experiment time
    """
    print("Running Epsilon-Specific iDLG Attacks on DP-Protected Data")
    print("Testing 5 clients per epsilon: round0 client0, round1 client1, ..., round4 client4")
    print("Using iDLG method only to save experiment time")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check attack data
    if not os.path.exists("idlg_inputs_dp"):
        print("No DP attack data found. Please run 'python server_dp.py' first.")
        return
    
    attack_files = [f for f in os.listdir("idlg_inputs_dp") if f.endswith('.pt')]
    if not attack_files:
        print("No DP attack data files found in idlg_inputs_dp/")
        return
    
    print(f"Found {len(attack_files)} DP attack data files")
    for f in sorted(attack_files):
        print(f"  - {f}")
    
    # Create attacker
    attacker = FixedIDLGAttackerDP(device)
    
    # Attack statistics
    total_attacks = 0
    successful_attacks = 0
    attack_results = []
    
    # Group files by epsilon
    epsilon_groups = {}
    for attack_file in attack_files:
        try:
            # Extract epsilon from filename (format: round{X}_client{Y}_eps{EPSILON}_dp_attack_data.pt)
            epsilon_str = attack_file.split('_eps')[1].split('_')[0]
            epsilon = float(epsilon_str)
            
            if epsilon not in epsilon_groups:
                epsilon_groups[epsilon] = []
            epsilon_groups[epsilon].append(attack_file)
        except:
            print(f"Warning: Could not extract epsilon from {attack_file}")
            continue
    
    print(f"\nFound {len(epsilon_groups)} different epsilon values:")
    for eps in sorted(epsilon_groups.keys(), reverse=True):
        print(f"  ε={eps}: {len(epsilon_groups[eps])} files")
    
    # Test each epsilon group - only iDLG method
    method = 'iDLG'
    
    for epsilon in sorted(epsilon_groups.keys(), reverse=True):
        files = sorted(epsilon_groups[epsilon])
        print(f"\n{'='*60}")
        print(f"TESTING EPSILON = {epsilon}")
        print(f"{'='*60}")
        print(f"Files to test: {len(files)}")
        
        epsilon_results = []
        epsilon_start_time = time.time()
        
        for attack_file in files:
            attack_path = os.path.join("idlg_inputs_dp", attack_file)
            print(f"\n--- Testing: {attack_file} ---")
            
            total_attacks += 1
            
            # Run iDLG attack with same parameters as original idlg_attack.py
            reconstructed, pred_label, results = attacker.attack(
                attack_path, 
                method=method, 
                lr=1.0,  # Same as original
                iterations=300  # Same as original
            )
            
            if reconstructed is not None and results is not None:
                successful_attacks += 1
                
                result_data = {
                    'file': attack_file,
                    'method': method,
                    'epsilon': epsilon,
                    'mse': results['mse'],
                    'psnr': results['psnr'],
                    'ssim': results['ssim'],
                    'time': results['time'],
                    'noise_multiplier': results['noise_multiplier'],
                    'final_loss': results['final_loss'],
                    'iterations': results['iterations']
                }
                attack_results.append(result_data)
                epsilon_results.append(result_data)
                
                print(f"iDLG attack completed successfully")
                print(f"MSE: {results['mse']:.6f}, PSNR: {results['psnr']:.2f}, Time: {results['time']:.2f}s")
            else:
                print(f"iDLG attack failed")
        
        # Epsilon-specific summary
        epsilon_total_time = time.time() - epsilon_start_time
        if epsilon_results:
            avg_time_per_attack = sum(r['time'] for r in epsilon_results) / len(epsilon_results)
            avg_mse = sum(r['mse'] for r in epsilon_results) / len(epsilon_results)
            
            print(f"\nEPSILON {epsilon} SUMMARY:")
            print(f"  Attacks: {len(epsilon_results)}/{len(files)} successful")
            print(f"  Average time per attack: {avg_time_per_attack:.2f}s")
            print(f"  Estimated time for 25 attacks: {avg_time_per_attack * 5:.2f}s")
            print(f"  Average MSE: {avg_mse:.6f}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total attacks performed: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    if total_attacks > 0:
        print(f"Success rate: {successful_attacks/total_attacks*100:.1f}%")
    
    if attack_results:
        # Group results by epsilon for comparison
        print(f"\nDETAILED RESULTS BY EPSILON:")
        print(f"{'='*50}")
        
        for epsilon in sorted(epsilon_groups.keys(), reverse=True):
            epsilon_results = [r for r in attack_results if r['epsilon'] == epsilon]
            if epsilon_results:
                avg_time = sum(r['time'] for r in epsilon_results) / len(epsilon_results)
                avg_mse = sum(r['mse'] for r in epsilon_results) / len(epsilon_results)
                avg_psnr = sum(r['psnr'] for r in epsilon_results) / len(epsilon_results)
                avg_ssim = sum(r['ssim'] for r in epsilon_results) / len(epsilon_results)
                
                # Privacy protection assessment
                critical_count = sum(1 for r in epsilon_results if r['mse'] < 0.01)
                protected_count = sum(1 for r in epsilon_results if r['mse'] >= 0.1)
                
                print(f"\nε = {epsilon}:")
                print(f"  Attacks: {len(epsilon_results)}")
                print(f"  Avg time: {avg_time:.2f}s (Est. for 25: {avg_time * 5:.2f}s)")
                print(f"  Avg MSE: {avg_mse:.6f}")
                print(f"  Avg PSNR: {avg_psnr:.2f} dB")
                print(f"  Avg SSIM: {avg_ssim:.4f}")
                print(f"  Critical breaches: {critical_count}/{len(epsilon_results)}")
                print(f"  Well protected: {protected_count}/{len(epsilon_results)}")
        
        # Time comparison guide
        print(f"\nTIME COMPARISON GUIDE:")
        print(f"To compare with standard iDLG attacks:")
        print(f"1. Run 'python idlg_attack.py' to get baseline times")
        print(f"2. Multiply DP attack times by 5 for fair comparison")
        print(f"3. Compare the scaled DP times with baseline times")
        
        # Privacy effectiveness conclusion
        all_mse = [r['mse'] for r in attack_results]
        critical_total = sum(1 for mse in all_mse if mse < 0.01)
        protected_total = sum(1 for mse in all_mse if mse >= 0.1)
        
        print(f"\nPRIVACY PROTECTION EFFECTIVENESS:")
        print(f"Critical privacy breaches: {critical_total}/{len(all_mse)} ({critical_total/len(all_mse)*100:.1f}%)")
        print(f"Well protected cases: {protected_total}/{len(all_mse)} ({protected_total/len(all_mse)*100:.1f}%)")
        
        if protected_total >= len(all_mse) // 2:
            conclusion = "Differential Privacy provides EFFECTIVE protection"
        elif critical_total >= len(all_mse) // 2:
            conclusion = "Differential Privacy provides LIMITED protection"
        else:
            conclusion = "Differential Privacy provides MODERATE protection"
        
        print(f"CONCLUSION: {conclusion}")
    
    print(f"\nExperiment completed! Results saved in 'idlg_results_dp/' directory")
    print(f"To compare attack times: multiply DP times by 5 and compare with baseline")

if __name__ == "__main__":
    run_epsilon_specific_idlg_attacks()