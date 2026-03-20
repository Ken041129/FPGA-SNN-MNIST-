# iverilog模擬:
iverilog -o tb_lif tb_lif.v lif.v  
vvp tb_lif  
gtkwave tb_lif.vcd
