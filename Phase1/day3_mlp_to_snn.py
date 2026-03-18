"""
Day 3：從 MLP 到 SNN — 把 ReLU 換成 LIF 神經元
==================================================
今天的目標：理解 MLP → SNN 只需要兩個改動：
  1. 把 ReLU 換成 snn.Leaky（LIF 神經元）
  2. 加一個時間迴圈（因為 LIF 有記憶，需要跑多個 time step）

我們會把 Day 2 的 MLP 和今天的 SNN 放在一起對比，
讓你清楚看到差異在哪。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils
import matplotlib.pyplot as plt
import numpy as np

print("=" * 55)
print("  Day 3：從 MLP 到 SNN")
print("=" * 55)

# ============================================
# Part 1：先回顧 MLP 的模型（Day 2）
# ============================================
print("\n--- Part 1：回顧 MLP vs SNN 的差異 ---")

print("""
  ┌──────────────────────────────────────────────────┐
  │  MLP（Day 2）          │  SNN（今天）            │
  ├──────────────────────────────────────────────────┤
  │  fc1 = Linear(784,128) │  fc1 = Linear(784,128)  │  ← 一樣！
  │  relu = ReLU()         │  lif1 = snn.Leaky(β)    │  ← 改這裡
  │  fc2 = Linear(128,10)  │  fc2 = Linear(128,10)   │  ← 一樣！
  │  （沒有）              │  lif2 = snn.Leaky(β)    │  ← 加這個
  ├──────────────────────────────────────────────────┤
  │  forward:              │  forward:                │
  │    x = fc1(input)      │    for t in NUM_STEPS:   │  ← 加時間迴圈
  │    x = relu(x)         │      cur = fc1(input)    │
  │    x = fc2(x)          │      spk, mem = lif1(cur)│  ← LIF 取代 ReLU
  │    return x            │      cur = fc2(spk)      │
  │                        │      spk, mem = lif2(cur)│
  │                        │    return spk_record     │
  └──────────────────────────────────────────────────┘

  關鍵差異：
  • ReLU：input 進來 → 立刻算出 output（沒有時間概念）
  • LIF： input 進來 → 膜電位慢慢累積 → 超過閾值才發 spike
         所以需要「跑很多個 time step」讓它累積
""")


# ============================================
# Part 2：載入資料（跟 Day 2 一樣）
# ============================================
print("--- Part 2：載入 MNIST ---")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),  # SNN 版本用 0~1 正規化就好
])

train_dataset = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"訓練集: {len(train_dataset)} 張，測試集: {len(test_dataset)} 張\n")


# ============================================
# Part 3：定義 SNN 模型
# ============================================
print("--- Part 3：定義 SNN 模型 ---")

# SNN 超參數
NUM_STEPS = 25    # 每張圖片跑 25 個時間步
BETA      = 0.95  # LIF 膜電位衰減率

class SpikingMNIST(nn.Module):
    """
    SNN 版本的 MNIST 分類器

    架構跟 Day 2 的 MLP 一模一樣：784 → 128 → 10
    只是把 ReLU 換成 LIF 神經元
    """

    def __init__(self):
        super().__init__()

        # === 全連接層（跟 MLP 完全相同）===
        self.fc1 = nn.Linear(784, 128, bias=False)  # bias=False 簡化硬體
        self.fc2 = nn.Linear(128, 10,  bias=False)

        # === LIF 神經元（取代 ReLU）===
        # beta = 0.95 表示每個 time step，膜電位保留 95%、洩漏 5%
        self.lif1 = snn.Leaky(beta=BETA)
        self.lif2 = snn.Leaky(beta=BETA)

    def forward(self, x):
        """
        x: [batch, 1, 28, 28]

        MLP 的 forward 只跑一次，
        SNN 的 forward 要跑 NUM_STEPS 次（時間展開）
        """
        x_flat = x.view(x.size(0), -1)  # [batch, 784]

        # 初始化膜電位（LIF 的「記憶」，一開始是 0）
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # 用來記錄每個 time step 的 spike
        spk2_record = []

        # === 時間迴圈：這是 SNN 跟 MLP 最大的差異 ===
        for step in range(NUM_STEPS):

            # 第一層：全連接 → LIF
            cur1 = self.fc1(x_flat)             # 突觸電流 = W1 × input
            spk1, mem1 = self.lif1(cur1, mem1)  # LIF 更新膜電位，可能發 spike
            # spk1 是 0 或 1 的 tensor，形狀 [batch, 128]
            # mem1 是更新後的膜電位

            # 第二層：全連接 → LIF
            cur2 = self.fc2(spk1)               # 注意：輸入是 spike（0 或 1）
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_record.append(spk2)

        # 把所有 time step 的 spike 疊起來
        # shape: [NUM_STEPS, batch, 10]
        return torch.stack(spk2_record)


model = SpikingMNIST()

total_params = sum(p.numel() for p in model.parameters())
print(f"模型架構: 784 → LIF(128) → LIF(10)")
print(f"總參數量: {total_params:,}")
print(f"時間步數: {NUM_STEPS}")
print(f"Beta:     {BETA}")
print(f"fc1 權重: {model.fc1.weight.shape} = {model.fc1.weight.numel():,}")
print(f"fc2 權重: {model.fc2.weight.shape}  = {model.fc2.weight.numel():,}")

# 💡 注意跟 Day 2 MLP 的差異：
#    MLP: 101,770 參數（有 bias）
#    SNN: 101,632 參數（沒有 bias，省了 138 個參數）
#    拿掉 bias 是因為 FPGA 上少一個加法器 = 省資源


# ============================================
# Part 4：理解一個 LIF 神經元在做什麼
# ============================================
print("\n--- Part 4：觀察單一 LIF 神經元的行為 ---")

# 建一個獨立的 LIF 來觀察
demo_lif = snn.Leaky(beta=BETA)
mem = demo_lif.init_leaky()

# 模擬：每個 time step 注入固定電流 0.3
input_current = 0.3
mem_history = []
spk_history = []

for t in range(NUM_STEPS):
    cur = torch.tensor([input_current])
    spk, mem = demo_lif(cur, mem)
    mem_history.append(mem.item())
    spk_history.append(spk.item())

print(f"  輸入電流: {input_current}（每個 time step 都一樣）")
print(f"  Beta: {BETA}")
print(f"  膜電位變化:")
for t in range(NUM_STEPS):
    bar = "█" * int(mem_history[t] * 30)
    spike_marker = " ← SPIKE!" if spk_history[t] > 0 else ""
    print(f"    t={t:2d} | mem={mem_history[t]:.3f} |{bar}{spike_marker}")

spike_times = [t for t, s in enumerate(spk_history) if s > 0]
print(f"\n  Spike 發射時間: {spike_times}")
print(f"  共發射 {len(spike_times)} 次 spike（在 {NUM_STEPS} 個時間步內）")

# 💡 你可以看到膜電位一直在累積，到超過 1.0（閾值）時發 spike，
#    然後被 reset（減去閾值），接著又開始累積...
#    這就是 LIF 神經元的全部行為！
#    在 Verilog 裡：一個加法器 + 一個移位器（乘以 beta）+ 一個比較器


# ============================================
# Part 5：畫出 LIF 行為
# ============================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(mem_history, "b-", linewidth=1.5, label="Membrane potential")
ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.7, label="Threshold")
for t in spike_times:
    ax1.axvline(x=t, color="orange", alpha=0.3, linewidth=8)
ax1.set_ylabel("Membrane potential")
ax1.legend()
ax1.set_title(f"LIF Neuron (β={BETA}, input={input_current})")
ax1.grid(True, alpha=0.3)

# Raster plot（spike 的時間點）
ax2.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, color="darkorange")
ax2.set_xlim(ax1.get_xlim())
ax2.set_yticks([])
ax2.set_xlabel("Time step")
ax2.set_ylabel("Spikes")

plt.tight_layout()
plt.savefig("lif_behavior.png", dpi=100)
print("\n已儲存 lif_behavior.png（打開看看 LIF 的膜電位變化和 spike 時間點！）")


# ============================================
# Part 6：訓練 SNN！
# ============================================
print("\n--- Part 6：訓練 SNN ---")

# 損失函數：MSE Spike Count Loss
# 正確類別目標 spike rate = 80%，錯誤類別 = 20%
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

NUM_EPOCHS = 10  # SNN 通常需要比 MLP 多訓練幾輪

train_losses = []
test_accs = []

print(f"開始訓練（{NUM_EPOCHS} epochs，每個 epoch 約 30-60 秒）...\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    batch_count = 0

    for images, labels in train_loader:

        # 前向傳播（跟 MLP 一樣，只是 model 內部有時間迴圈）
        spk_rec = model(images)  # [NUM_STEPS, batch, 10]

        # 計算損失
        loss = loss_fn(spk_rec, labels)

        # 反向傳播（surrogate gradient 在背後自動處理）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    # 測試準確率
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            spk_rec = model(images)

            # Rate coding 解碼：數哪個輸出神經元 spike 最多
            spike_counts = spk_rec.sum(dim=0)  # [batch, 10]
            predicted = spike_counts.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = epoch_loss / batch_count
    train_losses.append(avg_loss)
    test_accs.append(accuracy)

    print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Test Acc: {accuracy*100:.2f}%")


# ============================================
# Part 7：跟 MLP 比較
# ============================================
print("\n--- Part 7：SNN vs MLP 比較 ---")

mlp_acc = 96.30  # Day 2 的結果
snn_acc = test_accs[-1] * 100

print(f"""
  ┌───────────────────────────────────────┐
  │           MLP        SNN              │
  │  準確率   {mlp_acc:.1f}%     {snn_acc:.1f}%           │
  │  參數量   101,770    101,632           │
  │  激活函數 ReLU       LIF              │
  │  時間步   1          {NUM_STEPS}               │
  │  輸出方式 分數最高   spike 次數最多    │
  └───────────────────────────────────────┘

  SNN 的準確率通常比 MLP 低 1-4%，這是正常的。
  但 SNN 的硬體優勢是巨大的：
  • spike 是 0 或 1 → 第二層的乘法變成「選擇性加法」
  • 大部分 spike 是 0 → 事件驅動，不需要運算 → 超省電
  • 沒有乘法器（beta 可以用移位近似）→ FPGA 面積小
""")


# ============================================
# Part 8：觀察 spike 的稀疏性
# ============================================
print("--- Part 8：觀察 spike 稀疏性 ---")

model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_loader))
    spk_rec = model(test_images)  # [25, 128, 10]

    # 算一下平均 spike rate
    avg_spike_rate = spk_rec.float().mean().item()
    print(f"  輸出層平均 spike rate: {avg_spike_rate:.3f}")
    print(f"  = 每個神經元每個 time step 只有 {avg_spike_rate*100:.1f}% 的機率發 spike")
    print(f"  → 超過 {(1-avg_spike_rate)*100:.0f}% 的運算可以跳過！這就是 SNN 省電的原因")

# 畫一張 spike raster plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5))

# 拿第一張測試圖片的 spike pattern
single_spk = spk_rec[:, 0, :].cpu().numpy()  # [25, 10]

axes[0].imshow(test_images[0].squeeze(), cmap="gray")
axes[0].set_title(f"Input image (label: {test_labels[0].item()})")
axes[0].axis("off")

# Spike raster plot
for neuron in range(10):
    spike_times = np.where(single_spk[:, neuron] > 0)[0]
    axes[1].eventplot(spike_times, lineoffsets=neuron, linelengths=0.8,
                      color="darkorange" if neuron == test_labels[0].item() else "steelblue")

axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Output neuron")
axes[1].set_yticks(range(10))
axes[1].set_title("Output spike raster (orange = correct class)")
axes[1].set_xlim(-0.5, NUM_STEPS - 0.5)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("snn_spike_raster.png", dpi=100)
print("\n已儲存 snn_spike_raster.png（看看哪個神經元 spike 最多！）")


# ============================================
# Part 9：儲存訓練曲線
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-o", markersize=4)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("SNN Training Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in test_accs], "r-o", markersize=4)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("SNN Test Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("snn_training_curves.png", dpi=100)
print("已儲存 snn_training_curves.png\n")


# ============================================
# 小結
# ============================================
print("=" * 55)
print("  Day 3 完成！")
print("=" * 55)
print(f"""
  你今天做到了：
  1. 把 MLP 的 ReLU 換成 LIF 神經元 → 變成 SNN
  2. 理解時間迴圈：SNN 用 {NUM_STEPS} 個 time step 處理一張圖片
  3. 觀察 LIF 的膜電位累積 → 超過閾值 → 發 spike → reset
  4. 訓練 SNN 達到 {snn_acc:.1f}% 準確率
  5. 看到 spike 的稀疏性：>{(1-avg_spike_rate)*100:.0f}% 的運算可以跳過

  🔗 跟 FPGA 的連結：
  • fc1, fc2 的權重 → Block RAM
  • LIF 神經元 → 加法器 + 移位器 + 比較器
  • spike（0 或 1）→ 1-bit 訊號線
  • 時間迴圈 → FPGA 上的 clock cycle
  • spike 稀疏性 → 事件驅動架構可以省電

  接下來 Day 4-5：
  我們會動手寫一個純 Python 的 LIF（不用 snnTorch），
  確保你完全理解它的每一步運算，
  這直接對應到你後面要寫的 Verilog 模組。
""")
