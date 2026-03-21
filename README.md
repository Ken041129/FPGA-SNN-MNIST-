# Phase 1  
學 PyTorch，訓練 SNN 到 96%，手寫定點數 LIF，匯出 hex 檔。  
Day 1：tensor 和矩陣乘法 — 神經網路的基本運算  
Day 2：完整的 MLP 分類器 — 96.3% 準確率  
Day 3：把 ReLU 換成 LIF — 變成 SNN，96.1%  
Day 4：手寫定點數 LIF — 跟 snnTorch 結果一致。  
Day 5：匯出 .hex + 測試向量 — FPGA 可以直接讀取  
# Phase 2  
用 Verilog 寫了 lif.v → mac.v → fc_layer.v → snn_top.v，全部通過驗證，最終跟 Python 的 golden reference bit-for-bit 一致。
