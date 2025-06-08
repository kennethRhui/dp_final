import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CNN
from utils import calculate_psnr, calculate_ssim

def run_idlg_attack_he(gradient_path, data_path, device=torch.device('cpu')):
    """
    在 HE 保護的梯度上運行 iDLG 攻擊
    """
    try:
        # 載入解密後的聚合梯度
        if not os.path.exists(gradient_path):
            print(f"Gradient file not found: {gradient_path}")
            return None, None, 0.0, 0.0
        
        target_gradients = torch.load(gradient_path, map_location='cpu')  # 強制載入到 CPU
        
        # 載入原始數據用於比較
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return None, None, 0.0, 0.0
        
        data_info = torch.load(data_path, map_location='cpu')  # 強制載入到 CPU
        original_image = data_info['image'].unsqueeze(0)  # 添加批次維度，保持在 CPU
        true_label = data_info['label'].item()
        
        print(f"Loaded data - Image shape: {original_image.shape}, Label: {true_label}")
        
        # 將數據移動到指定設備進行計算
        original_image_device = original_image.to(device)
        target_gradients_device = [grad.to(device) for grad in target_gradients]
        
        # 初始化虛擬模型（用於計算梯度）
        net = CNN().to(device)
        net.eval()
        
        # 從目標梯度預測標籤
        predicted_label = predict_label_from_gradient(target_gradients)
        print(f"Predicted label from gradient: {predicted_label}, True label: {true_label}")
        
        # 初始化虛擬圖像（隨機初始化）
        dummy_image = torch.randn_like(original_image_device, device=device, requires_grad=True)
        dummy_label = torch.tensor([predicted_label], device=device)
        
        # 使用 Adam 優化器進行初步優化
        print("Starting Adam optimization for 1000 iterations...")
        optimizer_adam = torch.optim.Adam([dummy_image], lr=0.1)
        
        for iteration in range(1000):
            optimizer_adam.zero_grad()
            
            # 確保模型參數需要梯度
            for param in net.parameters():
                param.requires_grad = True
            
            # 計算虛擬梯度
            dummy_output = net(dummy_image)
            dummy_loss = F.cross_entropy(dummy_output, dummy_label)
            dummy_gradients = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )
            
            # 計算梯度差異損失
            grad_diff = 0
            for dummy_grad, target_grad in zip(dummy_gradients, target_gradients_device):
                grad_diff += ((dummy_grad - target_grad) ** 2).sum()
            
            grad_diff.backward()
            optimizer_adam.step()
            
            # 約束圖像值在合理範圍內
            with torch.no_grad():
                dummy_image.data = torch.clamp(dummy_image.data, 0, 1)
            
            if iteration % 200 == 0:
                print(f"Adam Iteration: {iteration}, Grad Diff Loss: {grad_diff.item():.6f}")
        
        # 使用 L-BFGS 進行精細優化
        print("Starting L-BFGS optimization for 300 iterations...")
        
        # 重新創建需要梯度的虛擬圖像
        dummy_image = dummy_image.detach().clone().requires_grad_(True)
        optimizer_lbfgs = torch.optim.LBFGS([dummy_image], max_iter=1)
        
        def closure():
            optimizer_lbfgs.zero_grad()
            
            # 重新初始化模型並確保參數需要梯度
            net.zero_grad()
            for param in net.parameters():
                param.requires_grad = True
            
            # 計算虛擬梯度
            dummy_output = net(dummy_image)
            dummy_loss = F.cross_entropy(dummy_output, dummy_label)
            
            # 使用 retain_graph=True 確保計算圖不被銷毀
            dummy_gradients = torch.autograd.grad(
                dummy_loss, 
                net.parameters(), 
                create_graph=True,
                retain_graph=True
            )
            
            # 計算梯度差異損失
            grad_diff = 0
            for dummy_grad, target_grad in zip(dummy_gradients, target_gradients_device):
                grad_diff += ((dummy_grad - target_grad) ** 2).sum()
            
            grad_diff.backward(retain_graph=True)
            
            # 約束圖像值
            with torch.no_grad():
                dummy_image.data = torch.clamp(dummy_image.data, 0, 1)
            
            return grad_diff
        
        # 執行 L-BFGS 優化
        for iteration in range(300):
            try:
                current_loss = optimizer_lbfgs.step(closure)
                if iteration % 50 == 0:
                    print(f"L-BFGS Iteration: {iteration}, Loss: {current_loss.item():.6f}")
            except Exception as e:
                print(f"L-BFGS optimization stopped at iteration {iteration}: {e}")
                break
        
        # 計算重建品質 - 確保所有計算都在 CPU 上進行
        with torch.no_grad():
            # 將結果移回 CPU 並確保正確的形狀
            reconstructed_img_cpu = dummy_image.squeeze(0).detach().cpu()
            original_img_cpu = original_image.squeeze(0)  # 原本就在 CPU 上
            
            # 確保數值範圍正確
            reconstructed_img_cpu = torch.clamp(reconstructed_img_cpu, 0, 1)
            original_img_cpu = torch.clamp(original_img_cpu, 0, 1)
            
            print(f"Computing metrics - Reconstructed shape: {reconstructed_img_cpu.shape}, Original shape: {original_img_cpu.shape}")
            
            # 計算指標
            psnr_val = calculate_psnr(original_img_cpu, reconstructed_img_cpu)
            ssim_val = calculate_ssim(original_img_cpu, reconstructed_img_cpu)
            
            print(f"Metrics computed - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        
        return reconstructed_img_cpu, predicted_label, psnr_val, ssim_val
        
    except Exception as e:
        print(f"Error in run_idlg_attack_he: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, 0.0

def predict_label_from_gradient(gradients):
    """
    從梯度預測標籤（使用最後一層的偏置梯度）
    """
    try:
        if len(gradients) >= 2:
            # 使用最後一層的偏置梯度
            last_bias_grad = gradients[-1]  # 形狀應該是 [10] 對於 10 個類別
            if last_bias_grad.numel() == 10:  # 確保是輸出層
                predicted_label = torch.argmax(last_bias_grad).item()
                return predicted_label
    except Exception as e:
        print(f"Error predicting label from gradient: {e}")
    
    # 如果無法從梯度預測，返回隨機標籤
    return np.random.randint(0, 10)

def save_comparison_he(original_img, reconstructed_img, save_path, psnr_val, ssim_val, round_num):
    """
    儲存原始圖像和重建圖像的比較
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # 確保圖像是正確的格式並在 CPU 上
        if isinstance(original_img, torch.Tensor):
            original_display = original_img.detach().cpu().numpy()
        else:
            original_display = np.array(original_img)
            
        if isinstance(reconstructed_img, torch.Tensor):
            reconstructed_display = reconstructed_img.detach().cpu().numpy()
        else:
            reconstructed_display = np.array(reconstructed_img)
        
        # 處理維度
        if len(original_display.shape) == 3 and original_display.shape[0] == 1:
            original_display = original_display.squeeze(0)
        if len(reconstructed_display.shape) == 3 and reconstructed_display.shape[0] == 1:
            reconstructed_display = reconstructed_display.squeeze(0)
        
        # 原始圖像
        axes[0].imshow(original_display, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 重建圖像
        axes[1].imshow(reconstructed_display, cmap='gray')
        axes[1].set_title(f'Reconstructed (HE)\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
        axes[1].axis('off')
        
        plt.suptitle(f'iDLG Attack Results - HE Round {round_num}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison image saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving comparison image: {e}")

def analyze_he_attack_results(he_results):
    """
    分析 HE 攻擊結果
    """
    if not he_results:
        print("No HE attack results to analyze.")
        return
    
    print("\n" + "="*60)
    print("HE ATTACK RESULTS ANALYSIS")
    print("="*60)
    
    total_psnr = 0
    total_ssim = 0
    successful_attacks = 0
    
    for round_num, (psnr, ssim, label_acc) in he_results.items():
        if psnr > 0:  # 只統計成功的攻擊
            total_psnr += psnr
            total_ssim += ssim
            successful_attacks += 1
            
            print(f"Round {round_num}:")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  SSIM: {ssim:.4f}")
            print(f"  Label Accuracy: {'✓' if label_acc else '✗'}")
            
            # 評估攻擊效果
            if psnr > 20 and ssim > 0.8:
                attack_quality = "High"
            elif psnr > 15 and ssim > 0.5:
                attack_quality = "Medium"
            else:
                attack_quality = "Low"
            print(f"  Attack Quality: {attack_quality}")
            print()
    
    if successful_attacks > 0:
        avg_psnr = total_psnr / successful_attacks
        avg_ssim = total_ssim / successful_attacks
        
        print(f"SUMMARY:")
        print(f"  Successful attacks: {successful_attacks}/{len(he_results)}")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        
        # HE 防護效果評估
        if avg_psnr < 15 and avg_ssim < 0.5:
            protection_level = "Strong"
        elif avg_psnr < 20 and avg_ssim < 0.7:
            protection_level = "Moderate"
        else:
            protection_level = "Weak"
        
        print(f"  HE Protection Level: {protection_level}")
        
        return avg_psnr, avg_ssim, protection_level
    else:
        print("All attacks failed - Strong HE protection detected!")
        return 0.0, 0.0, "Very Strong"

if __name__ == "__main__":
    print("Starting iDLG Attack on HE-Protected Gradients")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 創建輸出目錄
    os.makedirs("attack_results_he", exist_ok=True)
    
    # HE 攻擊結果
    he_results = {}
    
    # 對每個 HE 輪次進行攻擊
    for round_num in range(3):  # 3 輪 HE 訓練
        print(f"\n--- Attacking HE Round {round_num} ---")
        
        gradient_path = f"inputs_he/round{round_num}_client0_aggregated_decrypted_gradient_he.pt"
        data_path = f"inputs_he/round{round_num}_client0_data_he.pt"
        
        try:
            reconstructed_img, predicted_lbl, psnr_val, ssim_val = run_idlg_attack_he(
                gradient_path, data_path, device
            )
            
            if reconstructed_img is not None:
                # 載入原始圖像進行比較 - 確保在 CPU 上
                data_info = torch.load(data_path, map_location='cpu')
                original_image = data_info['image']
                true_label = data_info['label'].item()
                
                # 儲存比較圖像
                save_path = f"attack_results_he/he_round{round_num}_comparison.png"
                save_comparison_he(
                    original_image, reconstructed_img, save_path, 
                    psnr_val, ssim_val, round_num
                )
                
                # 檢查標籤準確性
                label_accuracy = (predicted_lbl == true_label)
                
                # 儲存結果
                he_results[round_num] = (psnr_val, ssim_val, label_accuracy)
                
                print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
                print(f"Label prediction: {predicted_lbl} (True: {true_label}) - {'Correct' if label_accuracy else 'Wrong'}")
                print(f"Results saved to {save_path}")
            else:
                print("Attack failed - could not reconstruct image")
                
        except Exception as e:
            print(f"Error during HE attack for round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 分析所有 HE 攻擊結果
    analyze_he_attack_results(he_results)