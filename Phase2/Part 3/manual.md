fc_layer.v 把 mac.v 和 lif.v 組合起來，處理一整層的推論。  
工作是：依序為每個神經元跑一次 MAC，然後把結果送進 LIF。  
iverilog -o tb_fc tb_fc_layer.v fc_layer.v mac.v
vvp tb_fc  
