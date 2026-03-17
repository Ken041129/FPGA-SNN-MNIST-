"""
Day 1：PyTorch Tensor 基礎
===========================
目標：理解 tensor 是什麼，為後面的 SNN 訓練做準備。

tensor 就是「多維陣列」，跟 NumPy 的 ndarray 幾乎一樣，
但多了兩個超能力：
  1. 可以跑在 GPU 上（我們用 CPU 也完全沒問題）
  2. 可以自動算梯度（訓練神經網路的核心）
"""

import torch

print("=" * 50)
print("  Day 1：PyTorch Tensor 基礎")
print("=" * 50)

# ============================================
# Part 1：建立 tensor
# ============================================
print("\n--- Part 1：建立 tensor ---")

# 從 Python list 建立
a = torch.tensor([1.0, 2.0, 3.0])
print(f"a = {a}")
print(f"a 的形狀: {a.shape}")      # torch.Size([3])
print(f"a 的資料型態: {a.dtype}")   # torch.float32

# 建立 2D tensor（矩陣）
b = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"\nb = \n{b}")
print(f"b 的形狀: {b.shape}")  # torch.Size([2, 3])  → 2 列 3 行

# 常用的初始化方式
zeros = torch.zeros(3, 4)       # 全 0 的 3x4 矩陣
ones  = torch.ones(2, 2)        # 全 1 的 2x2 矩陣
rand  = torch.randn(3, 3)       # 隨機（常態分佈）3x3 矩陣
print(f"\n全 0 矩陣:\n{zeros}")
print(f"隨機矩陣:\n{rand}")

# ============================================
# Part 2：tensor 運算
# ============================================
print("\n--- Part 2：tensor 運算 ---")

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 逐元素運算（跟 NumPy 一樣）
print(f"x + y = {x + y}")         # [5, 7, 9]
print(f"x * y = {x * y}")         # [4, 10, 18]  ← 逐元素相乘

# 矩陣乘法（這是神經網路的核心運算！）
W = torch.randn(2, 3)  # 2x3 的權重矩陣
result = W @ x          # 矩陣乘向量：(2x3) × (3,) = (2,)
print(f"\nW (2x3) @ x (3,) = {result}")
print(f"結果形狀: {result.shape}")  # torch.Size([2])

# 💡 這就是全連接層在做的事：output = W @ input
#    在我們的 SNN 中：
#    fc1: W 是 (128, 784)，input 是 (784,) → output 是 (128,)
#    fc2: W 是 (10, 128)，input 是 (128,) → output 是 (10,)

# ============================================
# Part 3：reshape（改變形狀）
# ============================================
print("\n--- Part 3：reshape ---")

# MNIST 圖片是 28x28，但全連接層需要一維向量
img = torch.randn(1, 28, 28)   # 模擬一張 MNIST 圖片
print(f"原始圖片形狀: {img.shape}")  # [1, 28, 28]

img_flat = img.view(-1)  # 攤平成一維
print(f"攤平後形狀:   {img_flat.shape}")  # [784]

# 💡 這就是我們 SNN 前向傳播的第一步：
#    x_flat = x.view(x.size(0), -1)  # batch 的圖片全部攤平

# ============================================
# Part 4：Autograd（自動微分）
# ============================================
print("\n--- Part 4：Autograd（自動微分） ---")

# 這是 PyTorch 最厲害的地方：自動幫你算梯度

# 建立一個「需要追蹤梯度」的 tensor
w = torch.tensor([2.0, 3.0], requires_grad=True)
x = torch.tensor([1.0, 4.0])

# 前向運算
y = w * x          # y = [2*1, 3*4] = [2, 12]
loss = y.sum()     # loss = 2 + 12 = 14

# 反向傳播：自動計算 dloss/dw
loss.backward()

print(f"w = {w.data}")
print(f"x = {x.data}")
print(f"y = w * x = {y.data}")
print(f"loss = sum(y) = {loss.item()}")
print(f"dloss/dw = {w.grad}")  # 應該是 [1, 4]，也就是 x 本身

# 💡 為什麼這很重要？
#    訓練神經網路 = 不斷調整 w 讓 loss 變小
#    要知道 w 該怎麼調，就需要 dloss/dw（梯度）
#    PyTorch 會自動幫你算，不用手推微分！

# ============================================
# Part 5：nn.Linear（全連接層）
# ============================================
print("\n--- Part 5：nn.Linear（全連接層） ---")

import torch.nn as nn

# nn.Linear(in, out) = 一個全連接層
# 內部自動建立了 W (out × in) 的權重矩陣
fc = nn.Linear(784, 128, bias=False)

print(f"全連接層 fc: 784 → 128")
print(f"權重形狀: {fc.weight.shape}")  # [128, 784]
print(f"權重總數: {fc.weight.numel():,}")  # 100,352

# 模擬一個 batch 的 MNIST 圖片通過這層
fake_batch = torch.randn(4, 784)  # 4 張攤平的圖片
output = fc(fake_batch)
print(f"\n輸入形狀:  {fake_batch.shape}")  # [4, 784]
print(f"輸出形狀:  {output.shape}")        # [4, 128]

# 💡 這就是我們 SNN 的 fc1 層！
#    784 個輸入（像素）→ 128 個輸出（隱藏神經元的突觸電流）

# ============================================
# 小結
# ============================================
print("\n" + "=" * 50)
print("  Day 1 完成！你學到了：")
print("=" * 50)
print("""
  1. tensor 是多維陣列，用法跟 NumPy 幾乎一樣
  2. 矩陣乘法 (W @ x) 是神經網路的核心運算
  3. .view(-1) 可以把 28x28 圖片攤平成 784 維向量
  4. requires_grad=True + .backward() = 自動算梯度
  5. nn.Linear(784, 128) = 一個有 100,352 個權重的全連接層

  明天我們會用這些概念組出一個完整的 MNIST MLP 分類器。
  跑完那個之後，再加入「時間」的概念就變成 SNN 了！
""")
