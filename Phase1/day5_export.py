"""
Day 5：匯出權重 + 測試向量 — Phase 1 收尾
=============================================
目標：
  1. 把 Day 4 的量化權重匯出成 .hex 檔案（Verilog 用 $readmemh 讀取）
  2. 產出測試向量（輸入圖片 + 預期輸出），供 Phase 2 testbench 驗證
  3. 產出一份完整的參數摘要，Phase 2 設計 RTL 時直接參考

跑完這個檔案，你的 fpga_export/ 資料夾就是 Phase 2 的起點。
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json

print("=" * 55)
print("  Day 5：匯出權重 + 測試向量")
print("=" * 55)

# ============================================
# 準備：訓練 SNN（跟 Day 3-4 一樣）
# ============================================
print("\n--- 訓練 SNN ---")

NUM_STEPS = 25
BETA = 0.95
THRESHOLD = 1.0
FRAC_BITS = 5
SCALE = 2 ** FRAC_BITS  # 32
Q_MIN, Q_MAX = -128, 127

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

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        spk_rec = model(images)
        loss = loss_fn(spk_rec, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                spk_rec = model(images)
                pred = spk_rec.sum(0).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        print(f"  Epoch {epoch+1:2d} | Test Acc: {correct/total*100:.2f}%")

# 取出權重
W1 = model.fc1.weight.data.cpu().numpy()
W2 = model.fc2.weight.data.cpu().numpy()

# 量化
W1_q = np.clip(np.round(W1 * SCALE), Q_MIN, Q_MAX).astype(np.int8)
W2_q = np.clip(np.round(W2 * SCALE), Q_MIN, Q_MAX).astype(np.int8)
beta_q = int(round(BETA * SCALE))
threshold_q = SCALE

print(f"\n訓練 + 量化完成！")


# ============================================
# Part 1：建立輸出目錄
# ============================================
OUT_DIR = "fpga_export"
TV_DIR  = os.path.join(OUT_DIR, "test_vectors")
os.makedirs(TV_DIR, exist_ok=True)

print(f"\n--- Part 1：輸出目錄 → {OUT_DIR}/ ---")


# ============================================
# Part 2：匯出 .hex 權重檔案
# ============================================
print("\n--- Part 2：匯出 .hex 權重檔案 ---")

def to_hex(val, bits=8):
    """有號整數 → 二補數十六進位字串"""
    if val < 0:
        val = (1 << bits) + val
    return format(val, f"0{bits // 4}X")


def export_weight_hex(name, weight_array, filepath):
    """
    匯出權重矩陣為 .hex 檔案

    格式：每行一個值，8-bit 二補數十六進位
    排列順序：逐行掃描（row-major）
      weight[0][0], weight[0][1], ..., weight[0][N-1],
      weight[1][0], weight[1][1], ..., weight[1][N-1],
      ...

    Verilog 讀取方式：
      reg signed [7:0] weight_mem [0:DEPTH-1];
      initial $readmemh("fc1_weight.hex", weight_mem);
    """
    rows, cols = weight_array.shape
    total = rows * cols

    with open(filepath, "w") as f:
        # 標頭註解（Verilog 會忽略 // 開頭的行）
        f.write(f"// {name}\n")
        f.write(f"// Shape: [{rows}, {cols}], Total: {total}\n")
        f.write(f"// Format: 8-bit signed Q2.5 (two's complement hex)\n")
        f.write(f"// Layout: row-major, weight[row][col]\n")
        f.write(f"// Verilog: $readmemh(\"{os.path.basename(filepath)}\", mem);\n")
        f.write(f"//\n")

        for i in range(rows):
            f.write(f"// --- neuron {i} (row {i}) ---\n")
            for j in range(cols):
                f.write(to_hex(int(weight_array[i, j])) + "\n")

    print(f"  ✅ {filepath}")
    print(f"     {rows}×{cols} = {total:,} 個值")
    print(f"     Range: [{weight_array.min()}, {weight_array.max()}]")
    print(f"     Verilog: reg signed [7:0] {name} [0:{total-1}];")


# 匯出 fc1 權重
export_weight_hex("fc1_weight",
                  W1_q,
                  os.path.join(OUT_DIR, "fc1_weight.hex"))

# 匯出 fc2 權重
export_weight_hex("fc2_weight",
                  W2_q,
                  os.path.join(OUT_DIR, "fc2_weight.hex"))

# 匯出常數
const_file = os.path.join(OUT_DIR, "constants.hex")
with open(const_file, "w") as f:
    f.write(f"// SNN Constants (8-bit hex)\n")
    f.write(f"// BETA = {BETA} → {beta_q} (0x{to_hex(beta_q)})\n")
    f.write(f"// THRESHOLD = {THRESHOLD} → {threshold_q} (0x{to_hex(threshold_q)})\n")
    f.write(f"{to_hex(beta_q)}\n")
    f.write(f"{to_hex(threshold_q)}\n")
print(f"  ✅ {const_file}")
print(f"     BETA = {BETA} → 0x{to_hex(beta_q)}")
print(f"     THRESHOLD = {THRESHOLD} → 0x{to_hex(threshold_q)}")


# ============================================
# Part 3：匯出測試向量
# ============================================
print("\n--- Part 3：匯出測試向量 ---")

def fixed_inference(image, w1_q, w2_q, beta_q, threshold_q, num_steps):
    """Day 4 的定點數推論（複製過來用）"""
    input_q = np.clip(np.round(image.flatten() * SCALE), 0, Q_MAX).astype(np.int32)
    mem1 = np.zeros(128, dtype=np.int32)
    mem2 = np.zeros(10,  dtype=np.int32)
    spk1_all, spk2_all = [], []

    for t in range(num_steps):
        cur1 = np.dot(w1_q.astype(np.int32), input_q) >> FRAC_BITS
        mem1 = (beta_q * mem1) >> FRAC_BITS
        mem1 = mem1 + cur1
        spk1 = (mem1 >= threshold_q).astype(np.int32)
        mem1 = mem1 - spk1 * threshold_q

        cur2 = np.dot(w2_q.astype(np.int32), spk1)
        mem2 = (beta_q * mem2) >> FRAC_BITS
        mem2 = mem2 + cur2
        spk2 = (mem2 >= threshold_q).astype(np.int32)
        mem2 = mem2 - spk2 * threshold_q

        spk1_all.append(spk1)
        spk2_all.append(spk2)

    counts = np.sum(spk2_all, axis=0)
    return np.argmax(counts), counts, spk1_all, spk2_all


num_test_vectors = 20
correct = 0

for idx in range(num_test_vectors):
    img_tensor, label = test_ds[idx]
    img = img_tensor.numpy().squeeze()

    pred, counts, spk1_all, spk2_all = fixed_inference(
        img, W1_q, W2_q, beta_q, threshold_q, NUM_STEPS
    )
    if pred == label:
        correct += 1

    # --- 匯出輸入圖片 ---
    input_file = os.path.join(TV_DIR, f"input_{idx:03d}.hex")
    input_q = np.clip(np.round(img.flatten() * SCALE), 0, Q_MAX).astype(np.int32)
    with open(input_file, "w") as f:
        f.write(f"// Test vector {idx}: label={label}, predicted={pred}\n")
        f.write(f"// 784 pixels, 8-bit unsigned (0~{Q_MAX})\n")
        for val in input_q:
            f.write(to_hex(int(val)) + "\n")

    # --- 匯出隱藏層 spike（用於中間層驗證）---
    spk1_file = os.path.join(TV_DIR, f"spk1_{idx:03d}.hex")
    with open(spk1_file, "w") as f:
        f.write(f"// Hidden layer spikes: {NUM_STEPS} timesteps × 128 neurons\n")
        f.write(f"// Each line = 128-bit vector packed as 32 hex chars\n")
        for t in range(NUM_STEPS):
            # 把 128 個 spike 打包成 128-bit hex
            bits = 0
            for j in range(128):
                bits |= (int(spk1_all[t][j]) << j)
            f.write(format(bits, "032X") + "\n")

    # --- 匯出輸出層 spike ---
    spk2_file = os.path.join(TV_DIR, f"spk2_{idx:03d}.hex")
    with open(spk2_file, "w") as f:
        f.write(f"// Output layer spikes: {NUM_STEPS} timesteps × 10 neurons\n")
        f.write(f"// Each line = 10-bit vector as 4 hex chars\n")
        for t in range(NUM_STEPS):
            bits = 0
            for j in range(10):
                bits |= (int(spk2_all[t][j]) << j)
            f.write(format(bits, "04X") + "\n")

    # --- 匯出預期分類結果 ---
    result_file = os.path.join(TV_DIR, f"result_{idx:03d}.txt")
    with open(result_file, "w") as f:
        f.write(f"label={label}\n")
        f.write(f"predicted={pred}\n")
        f.write(f"spike_counts={','.join(str(int(c)) for c in counts)}\n")

    status = "✅" if pred == label else "❌"
    print(f"  {status} Test {idx:2d}: label={label}, pred={pred}, "
          f"counts=[{' '.join(f'{int(c):2d}' for c in counts)}]")

print(f"\n  定點數準確率: {correct}/{num_test_vectors} "
      f"({correct/num_test_vectors*100:.0f}%)")


# ============================================
# Part 4：產出參數摘要（Phase 2 設計參考）
# ============================================
print("\n--- Part 4：產出參數摘要 ---")

summary = {
    "network": {
        "architecture": "784 → 128 → 10",
        "num_inputs": 784,
        "num_hidden": 128,
        "num_outputs": 10,
        "num_steps": NUM_STEPS,
    },
    "quantization": {
        "format": f"Q{8-FRAC_BITS-1}.{FRAC_BITS}",
        "total_bits": 8,
        "frac_bits": FRAC_BITS,
        "scale": SCALE,
        "range": f"[{Q_MIN}, {Q_MAX}]",
    },
    "lif_params": {
        "beta_float": BETA,
        "beta_fixed": beta_q,
        "beta_hex": f"0x{to_hex(beta_q)}",
        "threshold_float": THRESHOLD,
        "threshold_fixed": threshold_q,
        "threshold_hex": f"0x{to_hex(threshold_q)}",
    },
    "memory_requirements": {
        "fc1_weights": f"{784 * 128} bytes ({784 * 128 / 1024:.1f} KB)",
        "fc2_weights": f"{128 * 10} bytes ({128 * 10 / 1024:.1f} KB)",
        "total_weights": f"{784*128 + 128*10} bytes ({(784*128 + 128*10) / 1024:.1f} KB)",
        "mem1_state": "128 × 16-bit = 256 bytes",
        "mem2_state": "10 × 16-bit = 20 bytes",
    },
    "verilog_hints": {
        "weight_bitwidth": "signed [7:0]",
        "mem_bitwidth": "signed [15:0]  (needs wider for intermediate results)",
        "mac_bitwidth": "signed [23:0]  (8-bit × 8-bit accumulated 784 times)",
        "spike": "1-bit wire",
        "beta_multiply": f"mem * {beta_q} (constant multiplier, or shift-add approx)",
        "fc1_bram_depth": 784 * 128,
        "fc2_bram_depth": 128 * 10,
    }
}

summary_file = os.path.join(OUT_DIR, "design_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

# 也輸出一個人類可讀的版本
readme_file = os.path.join(OUT_DIR, "README.txt")
with open(readme_file, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  SNN MNIST FPGA Export — Phase 2 設計參考\n")
    f.write("=" * 60 + "\n\n")

    f.write("【網路架構】\n")
    f.write(f"  784 (input) → 128 (hidden, LIF) → 10 (output, LIF)\n")
    f.write(f"  時間步: {NUM_STEPS}\n\n")

    f.write("【量化格式】\n")
    f.write(f"  Q2.5 (8-bit signed)\n")
    f.write(f"  小數位: {FRAC_BITS}\n")
    f.write(f"  Scale factor: {SCALE}\n")
    f.write(f"  範圍: -4.0 ~ +3.96875\n\n")

    f.write("【LIF 參數】\n")
    f.write(f"  beta = {BETA} → 定點 {beta_q} (0x{to_hex(beta_q)})\n")
    f.write(f"  threshold = {THRESHOLD} → 定點 {threshold_q} (0x{to_hex(threshold_q)})\n\n")

    f.write("【Verilog 位寬建議】\n")
    f.write(f"  權重:     signed [7:0]   (8-bit)\n")
    f.write(f"  膜電位:   signed [15:0]  (16-bit，避免中間溢位)\n")
    f.write(f"  MAC累加:  signed [23:0]  (24-bit，784次累加不溢位)\n")
    f.write(f"  Spike:    1-bit\n\n")

    f.write("【記憶體需求】\n")
    f.write(f"  fc1 權重: {784*128:,} bytes = {784*128/1024:.1f} KB\n")
    f.write(f"  fc2 權重: {128*10:,} bytes = {128*10/1024:.1f} KB\n")
    f.write(f"  總計:     {(784*128+128*10)/1024:.1f} KB\n\n")

    f.write("【LIF 核心公式（Verilog 虛擬碼）】\n")
    f.write(f"  // 每個 clock cycle (= 1 time step):\n")
    f.write(f"  mem <= (BETA * mem) >>> {FRAC_BITS};   // 衰減\n")
    f.write(f"  mem <= mem + cur;                // 累加\n")
    f.write(f"  spk  = (mem >= THRESHOLD);       // 判斷\n")
    f.write(f"  if (spk) mem <= mem - THRESHOLD;  // reset\n\n")

    f.write("【檔案清單】\n")
    f.write(f"  fc1_weight.hex    — 第一層權重 (784×128)\n")
    f.write(f"  fc2_weight.hex    — 第二層權重 (128×10)\n")
    f.write(f"  constants.hex     — beta, threshold\n")
    f.write(f"  design_summary.json — 機器可讀的完整參數\n")
    f.write(f"  test_vectors/     — 測試向量 (input + expected spikes)\n")
    f.write(f"    input_XXX.hex   — 輸入圖片 (784 個 8-bit 值)\n")
    f.write(f"    spk1_XXX.hex    — 隱藏層預期 spike (128-bit × 25 steps)\n")
    f.write(f"    spk2_XXX.hex    — 輸出層預期 spike (10-bit × 25 steps)\n")
    f.write(f"    result_XXX.txt  — 預期分類結果\n")

print(f"  ✅ {summary_file}")
print(f"  ✅ {readme_file}")


# ============================================
# 最終確認：列出所有產出檔案
# ============================================
print("\n--- 最終確認：產出檔案清單 ---\n")

def list_files(directory, indent=0):
    total_size = 0
    for item in sorted(os.listdir(directory)):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print(f"{'  ' * indent}📁 {item}/")
            total_size += list_files(path, indent + 1)
        else:
            size = os.path.getsize(path)
            total_size += size
            if size > 1024:
                print(f"{'  ' * indent}📄 {item}  ({size/1024:.1f} KB)")
            else:
                print(f"{'  ' * indent}📄 {item}  ({size} bytes)")
    return total_size

total = list_files(OUT_DIR)
print(f"\n  總大小: {total/1024:.1f} KB")


# ============================================
# 小結
# ============================================
print("\n" + "=" * 55)
print("  🎉 Phase 1 完成！")
print("=" * 55)
print(f"""
  你在 Phase 1 做到的事：
  ──────────────────────
  Day 1: ✅ PyTorch tensor 基礎
  Day 2: ✅ 完整 MNIST MLP 分類器（96.3%）
  Day 3: ✅ 把 MLP 變成 SNN（96.1%）
  Day 4: ✅ 手寫定點數 LIF（99.0% on 200 samples）
  Day 5: ✅ 匯出 .hex 檔案 + 測試向量

  fpga_export/ 資料夾就是你 Phase 2 的起點。
  裡面有：
  • 權重 → Verilog 用 $readmemh 讀取
  • 測試向量 → Testbench 比對用
  • 設計參考 → 位寬、記憶體需求都算好了

  ────────────────────────────────────
  Phase 2 預告：用 Verilog 寫 LIF 神經元
  ────────────────────────────────────
  你的第一個 Verilog 模組會是一個 LIF neuron：
    - 輸入：cur (突觸電流)
    - 輸出：spk (spike)
    - 內部：mem (膜電位暫存器)

  就是 Day 4 那四行公式的硬體版本。
  準備好就跟我說「開始 Phase 2」！
""")
