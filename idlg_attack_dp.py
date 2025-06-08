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
        
    def attack(self, attack_data_path, method='iDLG', lr=0.1, iterations=300):
        """
        標準的 iDLG 攻擊實現
        """
        try:
            # 載入攻擊數據
            attack_data = torch.load(attack_data_path, map_location=self.device)
            
            # 提取必要資訊
            dp_gradients = [g.to(self.device) for g in attack_data['dp_gradients']]
            target_data = attack_data['target_data'].to(self.device)
            true_label = attack_data['true_label'].to(self.device)
            epsilon = attack_data['epsilon']
            noise_multiplier = attack_data['noise_multiplier']
            
            print(f"Loaded attack data: ε={epsilon}, noise_multiplier={noise_multiplier:.4f}")
            print(f"Target shape: {target_data.shape}, True label: {true_label.item()}")
            
            # 創建模型並初始化
            model = LeNetMNIST(channel=1, hidden=588, num_classes=10).to(self.device)
            
            # 簡單的參數初始化
            for param in model.parameters():
                param.data.normal_(0, 0.01)
            
            # iDLG 核心：從梯度推斷標籤
            if method == 'iDLG' and len(dp_gradients) > 0:
                # 對於分類問題，最後一層的偏置梯度可以推斷標籤
                last_bias_grad = dp_gradients[-1]  # 假設是偏置梯度
                inferred_label = torch.argmin(last_bias_grad).item()
                print(f"iDLG inferred label: {inferred_label}, True label: {true_label.item()}")
            else:
                inferred_label = torch.randint(0, 10, (1,)).item()
            
            # 初始化虛假數據和標籤
            dummy_data = torch.randn(target_data.shape, device=self.device, requires_grad=True)
            dummy_label = torch.tensor([inferred_label], device=self.device, dtype=torch.long)
            
            # 標準 LBFGS 優化器（iDLG 論文中使用的）
            optimizer = optim.LBFGS([dummy_data], lr=lr, max_iter=20)
            
            start_time = time.time()
            history = []
            
            print(f"Starting {method} reconstruction...")
            
            def closure():
                optimizer.zero_grad()
                
                # 前向傳播
                pred = model(dummy_data)
                loss = nn.CrossEntropyLoss()(pred, dummy_label)
                
                # 計算梯度
                gradients = torch.autograd.grad(
                    loss, model.parameters(), create_graph=True
                )
                
                # 計算與目標梯度的差異
                grad_diff = 0
                for grad, target_grad in zip(gradients, dp_gradients):
                    grad_diff += ((grad - target_grad) ** 2).sum()
                
                grad_diff.backward()
                return grad_diff
            
            # 優化過程
            for iteration in range(iterations // 20):  # LBFGS 每次內部迭代多次
                loss = optimizer.step(closure)
                
                current_loss = loss.item()
                history.append(current_loss)
                
                if iteration % 5 == 0:
                    print(f"Iteration {iteration*20}: Loss = {current_loss:.6f}")
                
                # 限制像素值
                with torch.no_grad():
                    dummy_data.clamp_(0, 1)
            
            attack_time = time.time() - start_time
            
            # 計算品質指標
            with torch.no_grad():
                reconstructed = dummy_data.cpu()
                target_cpu = target_data.cpu()
                
                mse = torch.mean((reconstructed - target_cpu) ** 2).item()
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                
                target_np = target_cpu.squeeze().numpy()
                reconstructed_np = reconstructed.squeeze().numpy()
                ssim_score = ssim(target_np, reconstructed_np, data_range=1.0)
            
            # 保存結果
            self.save_reconstruction_results(
                target_data.cpu(), reconstructed, true_label.item(), 
                inferred_label, attack_data_path, method, epsilon
            )
            
            results = {
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim_score,
                'time': attack_time,
                'epsilon': epsilon,
                'delta': attack_data.get('delta', 1e-5),
                'noise_multiplier': noise_multiplier,
                'iterations': iterations,
                'final_loss': history[-1] if history else float('inf')
            }
            
            print(f"{method} reconstruction completed!")
            print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_score:.4f}")
            
            return reconstructed, inferred_label, results
            
        except Exception as e:
            print(f"Error in {method} attack: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def save_reconstruction_results(self, original, reconstructed, true_label, pred_label, 
                                  attack_path, method, epsilon):
        """保存重建結果 - 與原版相同的格式"""
        try:
            os.makedirs('idlg_results_dp', exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(attack_path))[0]
            result_name = f"{base_name}_{method}_result.png"
            result_path = os.path.join('idlg_results_dp', result_name)
            
            # 計算MSE和其他指標用於標題
            mse = torch.mean((reconstructed - original) ** 2).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            target_np = original.squeeze().numpy()
            reconstructed_np = reconstructed.squeeze().numpy()
            ssim_score = ssim(target_np, reconstructed_np, data_range=1.0)
            
            # 計算絕對差異
            abs_diff = torch.abs(reconstructed - original)
            
            # 隱私評估
            if mse < 0.01:
                privacy_status = "CRITICAL PRIVACY BREACH"
            elif mse < 0.05:
                privacy_status = "SIGNIFICANT PRIVACY LEAK"
            elif mse < 0.1:
                privacy_status = "MODERATE PRIVACY RISK"
            else:
                privacy_status = "GOOD PRIVACY PROTECTION"
            
            # 創建三列圖像：原圖、重建圖、差異圖
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # 原始圖像
            axes[0].imshow(original.squeeze(), cmap='gray')
            axes[0].set_title(f'Original Private Data\nTrue Label: {true_label}')
            axes[0].axis('off')
            
            # 重建圖像
            axes[1].imshow(reconstructed.squeeze(), cmap='gray')
            axes[1].set_title(f'Fixed {method} Reconstructed\nPred Label: {pred_label}\nPSNR: {psnr:.2f} dB, SSIM: {ssim_score:.4f}')
            axes[1].axis('off')
            
            # 差異圖像
            im = axes[2].imshow(abs_diff.squeeze(), cmap='hot')
            axes[2].set_title(f'Absolute Difference\nMSE: {mse:.6f}')
            axes[2].axis('off')
            
            # 添加顏色條
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 添加總標題
            plt.suptitle(f'Fixed {method} Attack Results - {privacy_status}', 
                         fontsize=14, fontweight='bold', color='red' if 'BREACH' in privacy_status else 'black')
            
            plt.tight_layout()
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Results saved: {result_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

def run_multi_epsilon_dp_idlg_attacks():
    """
    運行多種epsilon值的差分隱私保護下的iDLG攻擊測試
    """
    print("Running Multi-Epsilon iDLG Attacks on Differential Privacy Protected Data")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 檢查攻擊數據
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
    
    # 創建攻擊器
    attacker = FixedIDLGAttackerDP(device)
    
    # 攻擊統計 - 與 idlg_results 相同的格式
    total_attacks = 0
    successful_attacks = 0
    attack_results = []
    
    # 測試所有文件
    for attack_file in sorted(attack_files):
        attack_path = os.path.join("idlg_inputs_dp", attack_file)
        print(f"\n{'='*50}")
        print(f"Testing: {attack_file}")
        print(f"{'='*50}")
        
        # 提取epsilon值
        try:
            epsilon_str = attack_file.split('_eps')[1].split('_')[0]
            epsilon = float(epsilon_str)
        except:
            epsilon = 1.0  # 默認值
        
        # 只測試 iDLG（與原版一致）
        method = 'iDLG'
        print(f"\n--- DP {method} Attack (ε={epsilon}) ---")
        total_attacks += 1
        
        reconstructed, pred_label, results = attacker.attack(
            attack_path, 
            method=method, 
            lr=1.0, 
            iterations=300
        )
        
        if reconstructed is not None and results is not None:
            successful_attacks += 1
            
            # 與 idlg_results 相同的數據結構
            result_data = {
                'file': attack_file,
                'method': method,
                'mse': results['mse'],
                'psnr': results['psnr'],
                'ssim': results['ssim'],
                'time': results['time'],
                'epsilon': results['epsilon'],
                'noise_multiplier': results['noise_multiplier'],
                'final_loss': results['final_loss']
            }
            attack_results.append(result_data)
            
            print(f"DP {method} attack completed successfully")
            print(f"MSE: {results['mse']:.6f}, PSNR: {results['psnr']:.2f}")
        else:
            print(f"DP {method} attack failed")
    
    # 與原版 idlg_results 完全相同的統計報告格式
    print(f"\n{'='*70}")
    print(f"MULTI-EPSILON DP-PROTECTED iDLG ATTACK SUMMARY")
    print(f"{'='*70}")
    print(f"Total attacks performed: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    if total_attacks > 0:
        print(f"Success rate: {successful_attacks/total_attacks*100:.1f}%")
    
    # 詳細結果分析 - 與原版格式一致
    if attack_results:
        print(f"\nDETAILED ATTACK RESULTS:")
        print(f"{'='*50}")
        
        # 按epsilon排序顯示結果
        sorted_results = sorted(attack_results, key=lambda x: x['epsilon'], reverse=True)
        
        for result in sorted_results:
            print(f"\nFile: {result['file']}")
            print(f"Epsilon: {result['epsilon']}")
            print(f"Noise Multiplier: {result['noise_multiplier']:.4f}")
            print(f"MSE: {result['mse']:.6f}")
            print(f"PSNR: {result['psnr']:.2f} dB")
            print(f"SSIM: {result['ssim']:.4f}")
            print(f"Attack Time: {result['time']:.2f}s")
            print(f"Final Loss: {result['final_loss']:.6f}")
            
            # 攻擊效果評估
            if result['mse'] < 0.01:
                effectiveness = "CRITICAL PRIVACY BREACH"
            elif result['mse'] < 0.05:
                effectiveness = "SIGNIFICANT PRIVACY LEAK"
            elif result['mse'] < 0.1:
                effectiveness = "MODERATE PRIVACY RISK"
            else:
                effectiveness = "GOOD PRIVACY PROTECTION"
            
            print(f"Privacy Assessment: {effectiveness}")
            print("-" * 40)
        
        # 總體統計
        avg_mse = sum(r['mse'] for r in attack_results) / len(attack_results)
        avg_psnr = sum(r['psnr'] for r in attack_results) / len(attack_results)
        avg_ssim = sum(r['ssim'] for r in attack_results) / len(attack_results)
        avg_time = sum(r['time'] for r in attack_results) / len(attack_results)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average Attack Time: {avg_time:.2f}s")
        
        # 隱私保護效果分類
        critical_count = sum(1 for r in attack_results if r['mse'] < 0.01)
        significant_count = sum(1 for r in attack_results if 0.01 <= r['mse'] < 0.05)
        moderate_count = sum(1 for r in attack_results if 0.05 <= r['mse'] < 0.1)
        protected_count = sum(1 for r in attack_results if r['mse'] >= 0.1)
        
        print(f"\nPRIVACY PROTECTION BREAKDOWN:")
        print(f"Critical Breaches: {critical_count}/{len(attack_results)} ({critical_count/len(attack_results)*100:.1f}%)")
        print(f"Significant Leaks: {significant_count}/{len(attack_results)} ({significant_count/len(attack_results)*100:.1f}%)")
        print(f"Moderate Risks: {moderate_count}/{len(attack_results)} ({moderate_count/len(attack_results)*100:.1f}%)")
        print(f"Well Protected: {protected_count}/{len(attack_results)} ({protected_count/len(attack_results)*100:.1f}%)")
        
        # 最終結論
        if protected_count >= len(attack_results) // 2:
            conclusion = "Differential Privacy provides EFFECTIVE protection against iDLG attacks"
        elif moderate_count + protected_count >= len(attack_results) // 2:
            conclusion = "Differential Privacy provides MODERATE protection against iDLG attacks"
        else:
            conclusion = "Differential Privacy provides LIMITED protection against iDLG attacks"
        
        print(f"\nCONCLUSION: {conclusion}")
    
    print(f"\nAll DP-protected iDLG attacks completed!")
    print(f"Results saved in 'idlg_results_dp/' directory")
    print(f"Compare with 'idlg_results/' directory for non-DP baseline")

if __name__ == "__main__":
    run_multi_epsilon_dp_idlg_attacks()