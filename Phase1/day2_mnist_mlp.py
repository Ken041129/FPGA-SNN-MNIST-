"""
Day 2：完整的 MNIST MLP 分類器
================================
目標：理解「載入資料 → 建模型 → 訓練 → 測試」的完整流程。

這個程式的結構跟我們最終的 SNN 版本幾乎一模一樣：
  MLP 版本：  fc1 → ReLU → fc2 → Softmax
  SNN 版本：  fc1 → LIF  → fc2 → LIF
                     ↑ 只有這裡不同！

所以今天學的東西，明天直接帶著走。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=" * 50)
print("  Day 2：MNIST MLP 分類器")
print("=" * 50)

# ============================================
# Part 1：載入 MNIST 資料集
# ============================================
print("\n--- Part 1：載入 MNIST 資料集 ---")

# transforms.Compose = 資料前處理流水線
# 1. ToTensor()：把圖片從 PIL Image 轉成 tensor，值域從 0~255 變成 0~1
# 2. Normalize((0.5,), (0.5,))：把 0~1 再轉成 -1~1（讓訓練更穩定）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 下載 MNIST（第一次會下載 ~60MB，之後從 ./data 讀取）
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

print(f"訓練集大小: {len(train_dataset)} 張圖片")
print(f"測試集大小: {len(test_dataset)} 張圖片")

# DataLoader：自動打包成 batch，並且打亂順序
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# 偷看一下資料長什麼樣
sample_img, sample_label = train_dataset[0]
print(f"\n一張圖片的形狀: {sample_img.shape}")   # [1, 28, 28]
print(f"對應的標籤: {sample_label}")              # 某個 0~9 的數字
print(f"像素值範圍: [{sample_img.min():.2f}, {sample_img.max():.2f}]")

# 💡 重點：每張圖片是 1×28×28 的 tensor
#    1 = 灰階通道，28×28 = 像素
#    我們要把它攤平成 784 維向量才能餵進全連接層


# ============================================
# Part 2：看看 MNIST 長什麼樣
# ============================================
print("\n--- Part 2：視覺化 MNIST ---")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("MNIST samples", fontsize=14)

for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap="gray")  # squeeze 去掉通道維度
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=100)
print("已儲存 mnist_samples.png（打開看看手寫數字長什麼樣！）")


# ============================================
# Part 3：定義 MLP 模型
# ============================================
print("\n--- Part 3：定義 MLP 模型 ---")


class MnistMLP(nn.Module):
    """
    最簡單的 MLP：784 → 128 → 10

    跟 Day 1 學的 nn.Linear 一樣，只是串了兩層 + ReLU
    """

    def __init__(self):
        super().__init__()  # 固定寫法，不用管

        # 兩個全連接層（跟我們 SNN 版本的架構完全一樣）
        self.fc1 = nn.Linear(784, 128)   # 第一層：784 → 128
        self.relu = nn.ReLU()            # 激活函數（SNN 中會換成 LIF）
        self.fc2 = nn.Linear(128, 10)    # 第二層：128 → 10

    def forward(self, x):
        """
        前向傳播：資料從輸入流到輸出

        x 的形狀: [batch_size, 1, 28, 28]
        """
        # Step 1：攤平圖片
        x = x.view(x.size(0), -1)  # [batch, 1, 28, 28] → [batch, 784]

        # Step 2：第一層全連接 + ReLU
        x = self.fc1(x)    # [batch, 784] → [batch, 128]
        x = self.relu(x)   # 把負值變成 0（非線性轉換）

        # Step 3：第二層全連接
        x = self.fc2(x)    # [batch, 128] → [batch, 10]

        # 輸出 10 個數字，代表 0~9 各自的「分數」
        # 分數最高的那個 = 模型的預測
        return x


model = MnistMLP()

# 數一下參數量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型架構: 784 → 128 → 10")
print(f"總參數量: {total_params:,}")
print(f"  fc1 權重: {model.fc1.weight.shape} = {model.fc1.weight.numel():,}")
print(f"  fc1 偏差: {model.fc1.bias.shape}   = {model.fc1.bias.numel():,}")
print(f"  fc2 權重: {model.fc2.weight.shape}  = {model.fc2.weight.numel():,}")
print(f"  fc2 偏差: {model.fc2.bias.shape}    = {model.fc2.bias.numel():,}")

# 💡 注意：MLP 有 bias（偏差），SNN 版本我們會拿掉它
#    因為在 FPGA 上少一個加法器 = 省資源


# ============================================
# Part 4：訓練！
# ============================================
print("\n--- Part 4：訓練模型 ---")

# 損失函數：CrossEntropyLoss
# 把模型的 10 個輸出分數跟正確答案比較，算出「錯多少」
loss_fn = nn.CrossEntropyLoss()

# 優化器：Adam
# 根據梯度自動調整權重，讓 loss 越來越小
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 5  # 整個訓練集跑 5 輪

train_losses = []
test_accs = []

for epoch in range(NUM_EPOCHS):
    model.train()  # 切換到訓練模式
    epoch_loss = 0
    batch_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: [128, 1, 28, 28] — 一個 batch 的圖片
        # labels: [128]             — 對應的正確答案

        # === 訓練三步驟（每個 batch 都重複） ===

        # Step 1：前向傳播 — 把圖片餵進模型，得到預測
        outputs = model(images)  # [128, 10]

        # Step 2：算 loss — 預測跟正確答案差多少
        loss = loss_fn(outputs, labels)

        # Step 3：反向傳播 + 更新權重
        optimizer.zero_grad()  # 清空上一次的梯度
        loss.backward()        # 算梯度（autograd 自動搞定）
        optimizer.step()       # 根據梯度更新權重

        epoch_loss += loss.item()
        batch_count += 1

    # --- 每個 epoch 結束後，測試一下準確率 ---
    model.eval()  # 切換到評估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 測試時不需要算梯度
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)  # 取分數最高的類別
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = epoch_loss / batch_count
    train_losses.append(avg_loss)
    test_accs.append(accuracy)

    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Test Acc: {accuracy*100:.2f}%")

# 💡 你應該會看到準確率從 ~90% 快速爬升到 ~97%
#    只用了 5 個 epoch！這是因為 MNIST 是個相對簡單的任務


# ============================================
# Part 5：看看模型學到了什麼
# ============================================
print("\n--- Part 5：看看預測結果 ---")

# 拿幾張測試圖片來看模型的預測
model.eval()
test_images, test_labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(test_images[:10])
    predictions = outputs.argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Model predictions", fontsize=14)

for i, ax in enumerate(axes.flat):
    img = test_images[i].squeeze()
    true_label = test_labels[i].item()
    pred_label = predictions[i].item()

    ax.imshow(img, cmap="gray")
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"True: {true_label}, Pred: {pred_label}", color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mlp_predictions.png", dpi=100)
print("已儲存 mlp_predictions.png（綠色=正確，紅色=錯誤）")


# ============================================
# Part 6：畫訓練曲線
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-o")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in test_accs], "r-o")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Test Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=100)
print("已儲存 training_curves.png")


# ============================================
# Part 7：看看權重長什麼樣
# ============================================
print("\n--- Part 7：視覺化 fc1 權重 ---")

# fc1 的每個輸出神經元都有 784 個權重
# 把這 784 個權重 reshape 回 28x28，就能看到這個神經元「在找什麼模式」
weights = model.fc1.weight.data

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
fig.suptitle("fc1 weight patterns (first 16 neurons)", fontsize=12)

for i, ax in enumerate(axes.flat):
    w = weights[i].view(28, 28)
    ax.imshow(w, cmap="seismic", vmin=-0.3, vmax=0.3)
    ax.axis("off")
    ax.set_title(f"#{i}", fontsize=8)

plt.tight_layout()
plt.savefig("fc1_weights.png", dpi=100)
print("已儲存 fc1_weights.png（每個方格 = 一個神經元學到的模式）")

# 💡 這些權重最終會被量化成 8-bit，存進 FPGA 的 Block RAM 裡
#    所以從 Python 到 Verilog，你處理的就是「這些數字」


# ============================================
# 小結
# ============================================
print("\n" + "=" * 50)
print("  Day 2 完成！你學到了：")
print("=" * 50)
print(f"""
  1. MNIST 載入 + DataLoader 打包成 batch
  2. 定義 MLP 模型 (784 → 128 → 10)
  3. 訓練三步驟：forward → loss → backward
  4. 達到 ~97% 準確率（只需要 5 個 epoch！）
  5. 權重視覺化：每個神經元都在找特定的模式

  📊 你的模型表現：
     最終準確率: {test_accs[-1]*100:.2f}%
     參數量: {total_params:,}

  🔗 跟 SNN 的連結：
     MLP:  fc1 → ReLU  → fc2 → Softmax（一次算完）
     SNN:  fc1 → LIF   → fc2 → LIF   （展開成多個時間步）

     Day 3 我們把 ReLU 換成 LIF 神經元，
     你會看到幾乎一樣的程式碼，只多了一個「時間迴圈」。
""")
