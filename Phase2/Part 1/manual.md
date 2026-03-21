tb_lif.v:  
Test 1：Reset 行為：確認按下重置鍵後，mem_out 是否確實回到 0 。  
Test 2：固定電流注入：這是最重要的測試。給予恆定電流 10 ，模擬 25 個時間步 ，觀察脈衝次數是否落在預期的 5~8 次之間 。  
Test 3：零輸入 (衰減測試)：先灌電流讓電位升高，然後切斷電流 (cur_in = 0) ，觀察電位是否隨時間慢慢漏掉（30 步後應小於 5） 。  
Test 4：致能控制 (Enable)：測試當 en = 0 時，就算給再大的電流，神經元也應該維持凍結狀態 。  
Test 5：負電流 (抑制性)：輸入負數電流 ，確認它不會發出脈衝 。  
指令:  
iverilog -o tb_lif tb_lif.v lif.v
vvp tb_lif  
看波型: gtkwave tb_lif.vcd
