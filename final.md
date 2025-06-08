# 🧠 Federated Learning Final Project

主題：使用 PyTorch + Flower 建立聯邦學習系統，並實作 iDLG 攻擊與差分隱私防禦

---

## ✅ 任務階段規劃總覽

1. 建立 Federated Learning 系統（Flower + PyTorch）
2. 實作 iDLG 攻擊（Improved Deep Leakage from Gradients）
3. 使用差分隱私進行防禦（Opacus）
4. （加分）實驗結果整合與報告撰寫

---

## 📍Step 1：建立 Federated Learning 系統

### 🎯 目標
- 使用 Flower 建構包含多個 Client 和一個 Server 的聯邦學習系統。
- 每個 Client 使用 PyTorch 訓練模型，資料為 MNIST 或 CIFAR10。

### 🔄 子任務
1. 建立 `client.py` 和 `server.py`：
   - 定義 CNN 模型（針對 MNIST）
   - 定義本地訓練與測試函式
   - 封裝成 Flower 的 `NumPyClient`
2. 將資料集切分為多位 client 的 local dataset（模擬 non-i.i.d）
3. 使用 FedAvg 做模型聚合
4. 執行訓練流程（10～20 輪），觀察模型收斂性
5. 儲存每輪的 local gradient，用作後續攻擊測試

### ✅ 輸出
- `client.py`, `server.py`, `model.py`
- 每輪訓練日誌與 accuracy
- 每輪 local gradient 檔案（如 `.pt`）

---

## 📍Step 2：實作 iDLG 攻擊

### 🎯 目標
- 還原 client 傳送的 gradient 對應的輸入圖片與標籤

### 🔄 子任務
1. 解析 Step 1 儲存的 gradients（可為 torch tensor）
2. 根據 iDLG 論文進行標籤還原：
   - 透過最後一層權重梯度方向確定 label（sign 分析）
3. 隨機初始化 dummy input `x_hat`，並固定 label
4. 最小化 `‖∇W(x_hat, label) − ∇W(real)‖²`，使用 L-BFGS 或 Adam
5. 記錄每輪重建的圖像
6. 評估還原品質：MSE、SSIM、視覺還原度

### ✅ 輸出
- `idlg_attack.py`
- 原圖與還原圖對照圖
- Fidelity 評分結果表（MSE, SSIM）

---

## 📍Step 3：使用 DP 防禦 iDLG 攻擊

### 🎯 目標
- 對 client 的 gradient 加入 noise，使 iDLG 攻擊失效

### 🔄 子任務
1. 使用 `Opacus` 將 local client optimizer 替換為 DP-SGD
2. 設定參數：
   - `clip_value`
   - `noise_multiplier`
   - 計算 ε 值範圍：0.1, 0.3, 0.5, 1.0, 5.0
3. 儲存每輪 DP 梯度，再次進行 iDLG 攻擊
4. 評估還原圖品質是否下降
5. 繪製 ε vs accuracy 與還原品質之折線圖

### ✅ 輸出
- `client_dp.py`
- DP 訓練結果（accuracy, ε）
- DP 梯度下攻擊結果與模糊圖
- ε vs accuracy 曲線圖

---

## 📍Step 4（加分）：整合報告與圖表分析

### 🎯 目標
- 視覺化整體實驗結果並撰寫報告與簡報

### 🔄 子任務
1. 建立 `report.md` 或 `report.docx` 文件
2. 匯出與統整：
   - FL 訓練過程 loss/accuracy 曲線
   - iDLG vs DP 還原圖對比
   - ε 值與準確率關係圖
3. 撰寫分析文字：
   - 各階段差異與隱私保護效果
   - 圖像是否成功保護？準確率是否下降？

### ✅ 輸出
- `report.md`（或 Word）
- 攻擊圖片總覽表
- PPT 簡報
- `TeamName_FinalProject.zip`

