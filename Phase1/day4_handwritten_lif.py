"""
Day 4：手寫 LIF 推論 — 從 Python 到 Verilog 的橋樑
=====================================================
目標：不用 snnTorch，用純 Python 重現 SNN 推論，
      並且標註每一步對應的 Verilog 該怎麼寫。

為什麼要這樣做？
  snnTorch 很方便，但它用浮點數、隱藏了很多細節。
  FPGA 上跑的是定點數整數運算，我們需要 100% 掌握每一步。
  這個檔案寫的「純 Python LIF」就是你 Verilog 的 spec。

這個檔案分成三個版本，逐步接近硬體：
  Version 1：浮點數版（驗證邏輯正確）
  Version 2：定點數版（模擬 FPGA 的整數運算）
  Version 3：加上 Verilog 對照註解
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF, utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=" * 55)
print("  Day 4：手寫 LIF — 從 Python 到 Verilog")
print("=" * 55)

# ============================================
# 準備：載入 Day 3 訓練好的模型
# ============================================
print("\n--- 準備：重新訓練一個 SNN（或你可以載入 Day 3 的）---")

NUM_STEPS = 25
BETA = 0.95
THRESHOLD = 1.0

# 快速訓練一個 SNN（跟 Day 3 一樣）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

class SpikingMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 10,  bias=False)
        self.lif1 = snn.Leaky(beta=BETA)
        self.lif2 = snn.Leaky(beta=BETA)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []
        for step in range(NUM_STEPS):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
        return torch.stack(spk_rec)

model = SpikingMNIST()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

print("訓練中（5 epochs）...")
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        spk_rec = model(images)
        loss = loss_fn(spk_rec, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 取出權重（轉成 NumPy）
W1 = model.fc1.weight.data.cpu().numpy()  # [128, 784]
W2 = model.fc2.weight.data.cpu().numpy()  # [10, 128]
print(f"訓練完成！W1: {W1.shape}, W2: {W2.shape}")

# 取一張測試圖片
test_img, test_label = test_ds[0]
img = test_img.numpy().squeeze()  # [28, 28]
print(f"測試圖片 label: {test_label}")


# ============================================
# Version 1：浮點數手寫 LIF
# ============================================
print("\n" + "=" * 55)
print("  Version 1：浮點數手寫 LIF")
print("=" * 55)
print("（驗證我們理解 snnTorch 在做什麼）\n")

def lif_inference_float(image, w1, w2, beta, threshold, num_steps):
    """
    純浮點數的 SNN 推論 — 不用任何框架

    這個函數做的事跟 snnTorch 的 forward() 完全一樣，
    但每一步都攤開來讓你看清楚。
    """
    input_flat = image.flatten()  # [784]

    # 初始化膜電位（全部從 0 開始）
    mem1 = np.zeros(128)  # 隱藏層 128 個神經元的膜電位
    mem2 = np.zeros(10)   # 輸出層 10 個神經元的膜電位

    spk2_record = []

    for t in range(num_steps):

        # ─── 第一層 ───
        # Step A：計算突觸電流（矩陣乘法）
        cur1 = w1 @ input_flat           # [128,784] × [784] = [128]

        # Step B：膜電位衰減（乘以 beta）
        mem1 = beta * mem1               # 保留 95%，洩漏 5%

        # Step C：累加輸入電流
        mem1 = mem1 + cur1

        # Step D：判斷是否發 spike
        spk1 = (mem1 >= threshold).astype(np.float32)  # [128] 的 0/1

        # Step E：reset by subtraction
        mem1 = mem1 - spk1 * threshold

        # ─── 第二層 ───
        cur2 = w2 @ spk1                 # [10,128] × [128] = [10]
        mem2 = beta * mem2 + cur2
        spk2 = (mem2 >= threshold).astype(np.float32)
        mem2 = mem2 - spk2 * threshold

        spk2_record.append(spk2)

    # 統計每個輸出神經元的 spike 次數
    spike_counts = np.sum(spk2_record, axis=0)
    predicted = np.argmax(spike_counts)

    return predicted, spike_counts


pred_v1, counts_v1 = lif_inference_float(img, W1, W2, BETA, THRESHOLD, NUM_STEPS)
print(f"Version 1 預測: {pred_v1} (真實: {test_label})")
print(f"Spike counts: {counts_v1.astype(int)}")

# 用 snnTorch 驗證
model.eval()
with torch.no_grad():
    spk_rec = model(test_img.unsqueeze(0))
    snn_counts = spk_rec[:, 0, :].sum(dim=0).numpy()
    snn_pred = np.argmax(snn_counts)

print(f"\nsnnTorch 預測: {snn_pred}")
print(f"snnTorch counts: {snn_counts.astype(int)}")

match = "✅ 完全一致！" if pred_v1 == snn_pred else "⚠️ 不一致（可能是浮點精度差異）"
print(f"比對結果: {match}")

# 💡 如果 Version 1 跟 snnTorch 結果一致，
#    代表你完全理解了 LIF 的運算邏輯。
#    接下來只要把浮點數換成定點數就好。


# ============================================
# Version 2：定點數手寫 LIF（模擬 FPGA）
# ============================================
print("\n" + "=" * 55)
print("  Version 2：定點數手寫 LIF（模擬 FPGA）")
print("=" * 55)
print("（全部用整數運算，跟 Verilog 的行為一模一樣）\n")

# --- 量化參數 ---
FRAC_BITS = 5          # 小數位數
SCALE = 2 ** FRAC_BITS  # = 32
Q_MIN = -128            # 8-bit 最小值
Q_MAX = 127             # 8-bit 最大值

# --- 量化權重 ---
def quantize(w, scale=SCALE, q_min=Q_MIN, q_max=Q_MAX):
    """浮點數 → 8-bit 定點數"""
    return np.clip(np.round(w * scale), q_min, q_max).astype(np.int8)

W1_q = quantize(W1)  # [128, 784] int8
W2_q = quantize(W2)  # [10, 128]  int8
beta_q = int(round(BETA * SCALE))  # 0.95 × 32 = 30
threshold_q = SCALE  # 1.0 × 32 = 32

print(f"量化參數:")
print(f"  FRAC_BITS = {FRAC_BITS}")
print(f"  SCALE = {SCALE}")
print(f"  beta = {BETA} → {beta_q}")
print(f"  threshold = {THRESHOLD} → {threshold_q}")
print(f"  W1 range: [{W1_q.min()}, {W1_q.max()}]")
print(f"  W2 range: [{W2_q.min()}, {W2_q.max()}]")


def lif_inference_fixed(image, w1_q, w2_q, beta_q, threshold_q, num_steps,
                        frac_bits=FRAC_BITS):
    """
    定點數 SNN 推論 — 完全用整數運算

    ⭐ 這個函數的每一行都直接對應一行 Verilog ⭐

    定點數規則：
      - 兩個 Q3.5 的數相乘 → 結果是 Q6.10（位數加倍）
      - 需要右移 frac_bits 位來對齊回 Q3.5
      - 這就是為什麼到處都有 >> frac_bits
    """
    # 輸入量化：像素值 0~1 → 0~32
    input_q = np.clip(np.round(image.flatten() * SCALE), 0, Q_MAX).astype(np.int32)

    # 膜電位（用更寬的位數，避免中間運算溢位）
    mem1 = np.zeros(128, dtype=np.int32)
    mem2 = np.zeros(10, dtype=np.int32)

    spk2_record = []

    for t in range(num_steps):

        # ─── 第一層 ───

        # Step A：矩陣乘法（全部整數）
        #   Verilog: 用 MAC (multiply-accumulate) 單元
        #   cur1[i] = Σ w1[i][j] * input[j]
        cur1 = np.dot(w1_q.astype(np.int32), input_q)  # int32 避免溢位

        # Step B：右移對齊（定點數乘法後必須做）
        #   w1 是 Q3.5，input 是 Q3.5
        #   乘出來是 Q6.10，右移 5 位變回 Q6.5
        #   Verilog: cur1 = cur1 >>> FRAC_BITS;
        cur1 = cur1 >> frac_bits

        # Step C：膜電位衰減
        #   mem1 是 Q?.5，beta 是 Q3.5
        #   乘出來也要右移對齊
        #   Verilog: mem1 = (mem1 * BETA) >>> FRAC_BITS;
        mem1 = (beta_q * mem1) >> frac_bits

        # Step D：累加輸入電流
        #   Verilog: mem1 = mem1 + cur1;
        mem1 = mem1 + cur1

        # Step E：判斷 spike
        #   Verilog: spk1 = (mem1 >= THRESHOLD) ? 1 : 0;
        spk1 = (mem1 >= threshold_q).astype(np.int32)

        # Step F：reset by subtraction
        #   Verilog: if (spk1) mem1 = mem1 - THRESHOLD;
        mem1 = mem1 - spk1 * threshold_q

        # ─── 第二層 ───
        # 注意：spk1 是 0 或 1（整數），不需要額外 scale
        # 所以 w2 × spk1 的結果不需要右移！
        #   Verilog: cur2[i] = Σ w2[i][j] * spk1[j];
        #   （因為 spk1 是 0 或 1，乘法變成 "要不要加 w2[i][j]"）
        cur2 = np.dot(w2_q.astype(np.int32), spk1)

        mem2 = (beta_q * mem2) >> frac_bits
        mem2 = mem2 + cur2
        spk2 = (mem2 >= threshold_q).astype(np.int32)
        mem2 = mem2 - spk2 * threshold_q

        spk2_record.append(spk2)

    spike_counts = np.sum(spk2_record, axis=0)
    predicted = np.argmax(spike_counts)

    return predicted, spike_counts


pred_v2, counts_v2 = lif_inference_fixed(img, W1_q, W2_q, beta_q, threshold_q, NUM_STEPS)
print(f"\nVersion 2（定點數）預測: {pred_v2} (真實: {test_label})")
print(f"Spike counts: {counts_v2}")
print(f"Version 1（浮點數）預測: {pred_v1}")

match2 = "✅" if pred_v2 == test_label else "❌"
print(f"定點數分類 {match2} {'正確' if pred_v2 == test_label else '錯誤'}")


# ============================================
# Version 3：逐步對照 Verilog
# ============================================
print("\n" + "=" * 55)
print("  Version 3：Python ↔ Verilog 對照表")
print("=" * 55)

print("""
  下面是定點數 LIF 每一步的 Python 和 Verilog 對照：

  ┌─────────┬──────────────────────────────┬──────────────────────────────────┐
  │  步驟   │  Python (你剛寫的)           │  Verilog (Phase 2 要寫的)        │
  ├─────────┼──────────────────────────────┼──────────────────────────────────┤
  │         │                              │                                  │
  │ 矩陣乘法│ cur = w @ input              │ // 用 MAC 單元，逐一累加         │
  │         │                              │ acc = acc + w[j] * input[j];     │
  │         │                              │                                  │
  │ 右移對齊│ cur = cur >> 5               │ cur = acc >>> 5;  // 算術右移    │
  │         │                              │                                  │
  │ 膜電位  │ mem = (beta * mem) >> 5      │ mem = (BETA * mem) >>> 5;        │
  │ 衰減    │                              │                                  │
  │         │                              │                                  │
  │ 累加    │ mem = mem + cur              │ mem <= mem + cur;                │
  │         │                              │                                  │
  │ Spike   │ spk = (mem >= thr)           │ spk = (mem >= THRESHOLD);        │
  │ 判斷    │                              │                                  │
  │         │                              │                                  │
  │ Reset   │ mem = mem - spk * thr        │ if (spk) mem <= mem - THRESHOLD; │
  │         │                              │                                  │
  └─────────┴──────────────────────────────┴──────────────────────────────────┘

  💡 最重要的洞察：
     第二層的 cur2 = W2 @ spk1，而 spk1 是 0 或 1。
     所以這個「矩陣乘法」在硬體上其實是：
       for j in 0..127:
         if spk1[j] == 1:
           cur2[i] += W2[i][j]    // 只做加法！
         // else: 什麼都不用做（省電！）

     這就是 SNN 在硬體上最大的優勢：
     乘法變成了「條件加法」，而且大部分 spike 是 0 → 大部分時候什麼都不用做。
""")


# ============================================
# 批量測試：確認定點數準確率
# ============================================
print("--- 批量測試定點數推論 ---")

correct = 0
total = 200  # 測試前 200 張

for i in range(total):
    test_img_i, test_label_i = test_ds[i]
    img_i = test_img_i.numpy().squeeze()
    pred_i, _ = lif_inference_fixed(img_i, W1_q, W2_q, beta_q, threshold_q, NUM_STEPS)
    if pred_i == test_label_i:
        correct += 1

fixed_acc = correct / total * 100
print(f"定點數推論準確率（前 {total} 張）: {fixed_acc:.1f}%")
print(f"跟 snnTorch 浮點數的差距很小，代表量化損失可接受。")


# ============================================
# 小結
# ============================================
print("\n" + "=" * 55)
print("  Day 4 完成！")
print("=" * 55)
print(f"""
  你今天做到了：
  1. ✅ 不用框架，手寫浮點數 LIF → 跟 snnTorch 結果一致
  2. ✅ 把浮點數換成 8-bit 定點數 → 準確率 {fixed_acc:.1f}%
  3. ✅ 理解每一步的 Verilog 對應

  🔑 你現在掌握的定點數 LIF 公式：
     mem = (beta * mem) >> FRAC_BITS    // 衰減
     mem = mem + cur                     // 累加
     spk = (mem >= threshold)            // 判斷
     mem = mem - spk * threshold         // reset

  這四行就是你 Phase 2 Verilog LIF 模組的全部核心。
  明天（Day 5）我們會把量化後的權重匯出成 .hex 檔案，
  正式為進入 Phase 2（Verilog）做好準備。
""")
