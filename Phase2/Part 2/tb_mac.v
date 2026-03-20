// =============================================================
// tb_mac.v — MAC 單元 Testbench
// =============================================================
//
// 測試策略：
//   用一個小的已知陣列（4 個元素），手算預期結果，
//   驗證 MAC 的累加和右移邏輯是否正確。
//
// 跑法：
//   iverilog -o tb_mac tb_mac.v mac.v
//   vvp tb_mac
//   gtkwave tb_mac.vcd
//
// =============================================================

`timescale 1ns / 1ps

module tb_mac;

    // ─── 用小陣列測試（4 個輸入，方便手算）───
    parameter NUM_INPUTS = 4;
    parameter W_WIDTH    = 8;
    parameter IN_WIDTH   = 8;
    parameter ACC_WIDTH  = 24;
    parameter OUT_WIDTH  = 16;
    parameter FRAC_BITS  = 5;
    parameter ADDR_WIDTH = 4;  // 2^4 = 16 > 4

    // ─── 信號 ───
    reg                          clk, rst_n, start;
    wire [ADDR_WIDTH-1:0]        weight_addr, input_addr;
    reg  signed [W_WIDTH-1:0]    weight_data;
    reg  signed [IN_WIDTH-1:0]   input_data;
    wire signed [OUT_WIDTH-1:0]  result;
    wire                         done;

    // ─── 實例化 MAC ───
    mac #(
        .NUM_INPUTS(NUM_INPUTS),
        .W_WIDTH(W_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .input_addr(input_addr),
        .input_data(input_data),
        .result(result),
        .done(done)
    );

    // ─── Clock ───
    initial clk = 0;
    always #5 clk = ~clk;

    // ─── 模擬 BRAM：用陣列代替 ───
    // 測試資料：
    //   weights = [3, -2, 5, 1]
    //   inputs  = [10, 20, 4, 8]
    //
    // 手算：
    //   acc = 3×10 + (-2)×20 + 5×4 + 1×8
    //       = 30   + (-40)   + 20  + 8
    //       = 18
    //   result = 18 >> 5 = 0（整數部分）
    //
    // 注意：18 >> 5 在算術右移中 = 0（因為 18 < 32）
    // 為了看到更明顯的結果，用更大的值試試

    // 改用更大的值：
    //   weights = [10, 20, -5, 15]
    //   inputs  = [30, 10, 20, 25]
    //
    //   acc = 10×30 + 20×10 + (-5)×20 + 15×25
    //       = 300   + 200   + (-100)  + 375
    //       = 775
    //   result = 775 >> 5 = 24（= 775 / 32 = 24.21875）

    reg signed [W_WIDTH-1:0]  weight_mem [0:3];
    reg signed [IN_WIDTH-1:0] input_mem  [0:3];

    initial begin
        weight_mem[0] =  10;
        weight_mem[1] =  20;
        weight_mem[2] =  -5;
        weight_mem[3] =  15;

        input_mem[0]  =  30;
        input_mem[1]  =  10;
        input_mem[2]  =  20;
        input_mem[3]  =  25;
    end

    // 模擬 1-cycle 讀取延遲的 BRAM
    always @(posedge clk) begin
        weight_data <= weight_mem[weight_addr];
        input_data  <= input_mem[input_addr];
    end

    // ─── VCD 波形 ───
    initial begin
        $dumpfile("tb_mac.vcd");
        $dumpvars(0, tb_mac);
    end

    // ─── 測試變數 ───
    integer test_pass, test_fail;
    integer expected_result;

    // ─── 主測試程序 ───
    initial begin
        $display("==================================================");
        $display("  MAC Unit Testbench");
        $display("==================================================");
        $display("  NUM_INPUTS=%0d, FRAC_BITS=%0d", NUM_INPUTS, FRAC_BITS);
        $display("");

        test_pass = 0;
        test_fail = 0;

        // ─── Reset ───
        rst_n = 0;
        start = 0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // ─── Test 1: 基本累加 ───
        $display("--- Test 1: Basic MAC operation ---");
        $display("  W = [10, 20, -5, 15]");
        $display("  X = [30, 10, 20, 25]");
        $display("  Expected: (10*30 + 20*10 + (-5)*20 + 15*25) >> 5");
        $display("          = (300 + 200 - 100 + 375) >> 5");
        $display("          = 775 >> 5 = 24");
        $display("");

        expected_result = 775 >>> FRAC_BITS;  // = 24

        // 啟動 MAC
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // 等待 done
        @(posedge done);
        @(posedge clk);  // 多等一個 cycle 讓 result 穩定
        #1;

        $display("  Result = %0d (expected %0d)", result, expected_result);

        if (result == expected_result) begin
            $display("  PASS!");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: got %0d, expected %0d", result, expected_result);
            test_fail = test_fail + 1;
        end

        // ─── Test 2: 連續執行兩次（確認 reset 正確）───
        $display("");
        $display("--- Test 2: Run twice (verify accumulator reset) ---");

        // 改變輸入值
        weight_mem[0] =  5;
        weight_mem[1] =  5;
        weight_mem[2] =  5;
        weight_mem[3] =  5;

        input_mem[0]  =  20;
        input_mem[1]  =  20;
        input_mem[2]  =  20;
        input_mem[3]  =  20;

        // 手算：5×20 × 4 = 400, 400 >> 5 = 12
        expected_result = 400 >>> FRAC_BITS;

        @(posedge clk);
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        @(posedge done);
        @(posedge clk);
        #1;

        $display("  W = [5, 5, 5, 5], X = [20, 20, 20, 20]");
        $display("  Expected: 400 >> 5 = %0d", expected_result);
        $display("  Result = %0d", result);

        if (result == expected_result) begin
            $display("  PASS: accumulator correctly reset between runs");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: got %0d (accumulator might not have reset)",
                     result);
            test_fail = test_fail + 1;
        end

        // ─── Test 3: 全零輸入 ───
        $display("");
        $display("--- Test 3: All-zero inputs ---");

        input_mem[0] = 0;
        input_mem[1] = 0;
        input_mem[2] = 0;
        input_mem[3] = 0;

        @(posedge clk);
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        @(posedge done);
        @(posedge clk);
        #1;

        if (result == 0) begin
            $display("  PASS: zero inputs → zero result");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: expected 0, got %0d", result);
            test_fail = test_fail + 1;
        end

        // ─── 結果摘要 ───
        $display("");
        $display("==================================================");
        $display("  Results: %0d PASS, %0d FAIL", test_pass, test_fail);
        $display("==================================================");

        if (test_fail == 0)
            $display("  All tests passed!");
        else
            $display("  WARNING: %0d test(s) failed!", test_fail);

        $display("");
        $finish;
    end

endmodule
