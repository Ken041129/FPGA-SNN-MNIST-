# Phase 3  
把設計搬上 Artix-7 FPGA。主要是加 UART 介面（讓 PC 傳圖片給 FPGA、FPGA 回傳預測結果），然後用 Vivado 合成佈局佈線。  
板子型號: Basys 3 (XC7A35T)  
## Vivado 操作步驟  
### 1. 建立新專案   
打開 Vivado → Create Project → Next  
Project name: 隨意  
Project location: 隨意  
Project type: RTL Project，勾選 Do not specify sources at this time
Part: 搜尋 xc7a35tcpg236-1  
Finish
### 2. 加入設計檔案  
在左側 Sources 面板 → 右鍵 Design Sources → Add Sources → Add or create design sources  
加入Phase 3資料夾中的 .v 檔案（全部）：  
fpga_top.v ← 設為 top module  
snn_top.v  
fc_layer.v  
mac.v  
lif.v  
uart_rx.v  
uart_tx.v  
### 3. 加入 constraints  
右鍵 Constraints → Add Sources → Add or create constraints  
加入 Basys3_SNN.xdc  
### 4. 加入權重 hex 檔  
把 fc1_weight.hex 和 fc2_weight.hex（要無註解）複製到 Vivado 專案目錄裡的 snn_mnist.srcs/sources_1/ 目錄下。或者在 Vivado 裡加為設計檔案。  
### 5. 跑合成（Synthesis）  
左側面板 → Run Synthesis  
* 跑完後可以去 Project Summary 裡看資源使用率（Utilization），可以注意 BRAM、LUT、DSP 這三項。  
### 6. 跑 Implementation、產生 Bitstream  
左側 Flow Navigator → Run Implementation → Generate Bitstream → Open Hardware Manager  
### 7. 燒 Bitstream 到 FPGA  
把 Basys 3 用 USB 接上電腦，然後在 Vivado 裡：  
Step 1： Flow Navigator → Open Hardware Manager → 點上方 Open Target → Auto Connect（Vivado 會偵測到 Basys 3） → 點 Program Device → 選剛產生的 .bit 檔 → Program  
Step 2： Windows 鍵 → 「裝置管理員」找 COM Port。  
Step 3： 安裝 pyserial 並跑測試: pip install pyserial  
打開 pc_uart_test.py，把第一行的 COM_PORT = "COM3" 改成你找到的 COM 號。然後確保測試圖片的 hex 檔在同一個目錄，跑：  
python pc_uart_test.py  
