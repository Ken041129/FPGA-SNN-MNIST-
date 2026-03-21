工作目錄/
├── snn_top.v
├── fc_layer.v
├── mac.v 
├── tb_snn_top.v
├── fc1_weight.hex   ← 從 fpga_export/ 複製
├── fc2_weight.hex   ← 從 fpga_export/ 複製
└── input_000.hex    ← 從 fpga_export/test_vectors/ 複製  
指令:  
iverilog -o tb_snn tb_snn_top.v snn_top.v fc_layer.v mac.v
vvp tb_snn  
# Debug  
.hex內註解要刪掉
