// =============================================================
// tb_lif.v — LIF 神經元 Testbench
// =============================================================
//
// 測試策略：
//   1. 注入固定電流，觀察膜電位累積和 spike 行為
//   2. 跟 Day 4 的 Python 參考模型比對
//   3. 測試邊界情況（reset, 零輸入, 負電流）
//
// 在 Vivado 中跑：
//   直接加入 Simulation Source，跑 behavioral simulation
//
// 或用 iverilog（免費）：
//   iverilog -o tb_lif tb_lif.v lif.v
//   vvp tb_lif
//
// =============================================================

`timescale 1ns / 1ps

module tb_lif;

    // ─── 參數 ───
    parameter MEM_WIDTH  = 16;
    parameter CUR_WIDTH  = 16;
    parameter FRAC_BITS  = 5;
    parameter BETA       = 30;     // 0.95 × 32
    parameter THRESHOLD  = 32;     // 1.0 × 32
    parameter NUM_STEPS  = 25;

    // ─── 信號宣告 ───
    reg                         clk;
    reg                         rst_n;
    reg                         en;
    reg  signed [CUR_WIDTH-1:0] cur_in;
    wire                        spk_out;
    wire signed [MEM_WIDTH-1:0] mem_out;

    // ─── 實例化 LIF 模組 ───
    lif #(
        .MEM_WIDTH(MEM_WIDTH),
        .CUR_WIDTH(CUR_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .BETA(BETA),
        .THRESHOLD(THRESHOLD)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .cur_in(cur_in),
        .spk_out(spk_out),
        .mem_out(mem_out)
    );

    // ─── Clock 產生（10ns 週期 = 100MHz）───
    initial clk = 0;
    always #5 clk = ~clk;

    // ─── 測試記錄 ───
    integer spike_count;
    integer test_pass;
    integer test_fail;
    integer t;

    // ─── Python 參考值 ───
    // Day 4 的結果：input=0.3 (=9.6→10 in fixed point), beta=0.95
    // Python spike times: [3, 7, 11, 15, 18, 22]
    //
    // 注意：Python 用 input=0.3×32=9.6→round→10
    // 但 Day 4 的 fixed_inference 是對整張圖片做矩陣乘法後的結果
    // 這裡我們用簡化的固定電流來測試 LIF 本身的邏輯

    // ─── VCD 波形輸出（給 GTKWave 用）───
    initial begin
        $dumpfile("tb_lif.vcd");
        $dumpvars(0, tb_lif);
    end

    // ─── 主測試程序 ───
    initial begin
        $display("==================================================");
        $display("  LIF Neuron Testbench");
        $display("==================================================");
        $display("  BETA=%0d, THRESHOLD=%0d, FRAC_BITS=%0d",
                 BETA, THRESHOLD, FRAC_BITS);
        $display("");

        test_pass = 0;
        test_fail = 0;

        // ─── Test 1：Reset 行為 ───
        $display("--- Test 1: Reset ---");
        rst_n = 0;
        en    = 0;
        cur_in = 0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        if (mem_out == 0) begin
            $display("  PASS: mem = 0 after reset");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: mem = %0d after reset (expected 0)", mem_out);
            test_fail = test_fail + 1;
        end

        // ─── Test 2：固定電流注入，觀察 spike ───
        $display("");
        $display("--- Test 2: Fixed current injection ---");
        $display("  Injecting cur_in = 10 (= 0.3125 in Q2.5)");
        $display("  Expected: membrane accumulates, spikes periodically");
        $display("");

        // Reset
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        en = 1;
        cur_in = 10;  // 10/32 = 0.3125，接近 Day 4 的 0.3

        spike_count = 0;

        $display("  step | mem_before | cur_in | mem_after | spike");
        $display("  -----+------------+--------+-----------+------");

        for (t = 0; t < NUM_STEPS; t = t + 1) begin
            @(posedge clk);
            #1;  // 等一下讓信號穩定

            if (spk_out) spike_count = spike_count + 1;

            $display("  %4d | %10d | %6d | %9d | %s",
                     t, mem_out, cur_in,
                     (spk_out) ? mem_out : mem_out,
                     (spk_out) ? "SPIKE!" : "      ");
        end

        $display("");
        $display("  Total spikes: %0d (in %0d steps)", spike_count, NUM_STEPS);

        // Day 4 Python 參考：0.3 的電流大約每 3-4 步 spike 一次
        // 0.3125 略大於 0.3，所以 spike 頻率應該略高
        if (spike_count >= 5 && spike_count <= 8) begin
            $display("  PASS: spike count in expected range [5, 8]");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: spike count %0d outside expected range", spike_count);
            test_fail = test_fail + 1;
        end

        // ─── Test 3：零輸入，膜電位應該逐漸衰減 ───
        $display("");
        $display("--- Test 3: Zero input (leak only) ---");

        // 先注入一些電流讓膜電位升高
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        en = 1;
        cur_in = 20;  // 較大電流
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);

        // 切到零輸入
        cur_in = 0;
        $display("  Membrane should decay towards 0:");

        for (t = 0; t < 30; t = t + 1) begin
            @(posedge clk);
            #1;
            if (t < 10 || t == 14 || t == 19 || t == 24 || t == 29)
                $display("  t=%0d: mem = %0d", t, mem_out);
        end

        // beta=30/32=0.9375, 從 ~20 開始衰減
        // 30 步後：20 × 0.9375^30 ≈ 2.8，應小於 5
        if (mem_out >= 0 && mem_out <= 5) begin
            $display("  PASS: membrane decayed near 0 (mem=%0d)", mem_out);
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: membrane = %0d (expected <= 5 after 30 steps)", mem_out);
            test_fail = test_fail + 1;
        end

        // ─── Test 4：en=0 時膜電位應保持不變 ───
        $display("");
        $display("--- Test 4: Enable control ---");

        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        en = 1;
        cur_in = 15;
        @(posedge clk);
        @(posedge clk);
        #1;

        begin : enable_test
            reg signed [MEM_WIDTH-1:0] mem_before_disable;
            mem_before_disable = mem_out;

            en = 0;  // 停止更新
            cur_in = 100;  // 即使電流很大也不應該影響
            @(posedge clk);
            @(posedge clk);
            @(posedge clk);
            #1;

            if (mem_out == mem_before_disable) begin
                $display("  PASS: membrane unchanged when en=0 (mem=%0d)", mem_out);
                test_pass = test_pass + 1;
            end else begin
                $display("  FAIL: membrane changed when en=0 (%0d -> %0d)",
                         mem_before_disable, mem_out);
                test_fail = test_fail + 1;
            end
        end

        // ─── Test 5：負電流（抑制性突觸）───
        $display("");
        $display("--- Test 5: Negative current (inhibitory) ---");

        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        en = 1;
        cur_in = -5;  // 負電流
        spike_count = 0;

        for (t = 0; t < 20; t = t + 1) begin
            @(posedge clk);
            #1;
            if (spk_out) spike_count = spike_count + 1;
        end

        if (spike_count == 0) begin
            $display("  PASS: no spikes with negative current (as expected)");
            test_pass = test_pass + 1;
        end else begin
            $display("  FAIL: %0d spikes with negative current", spike_count);
            test_fail = test_fail + 1;
        end

        // ─── 測試結果摘要 ───
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
