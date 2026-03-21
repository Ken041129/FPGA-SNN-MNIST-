// =============================================================
// tb_snn_top.v — SNN Top Module Testbench（修正版）
// =============================================================
//
// 需要的檔案（用 fix_hex_files.py 產生的乾淨版本）：
//   fc1_weight.hex   — 無註解，100352 行
//   fc2_weight.hex   — 無註解，1280 行
//   input_000.hex    — 無註解，784 行
//
// 跑法：
//   iverilog -o tb_snn tb_snn_top.v snn_top.v fc_layer.v mac.v
//   vvp tb_snn
//
// =============================================================

`timescale 1ns / 1ps

module tb_snn_top;

    parameter NUM_INPUTS  = 784;
    parameter NUM_HIDDEN  = 128;
    parameter NUM_OUTPUTS = 10;
    parameter NUM_STEPS   = 25;
    parameter W_WIDTH     = 8;
    parameter IN_WIDTH    = 8;
    parameter MEM_WIDTH   = 16;
    parameter ACC_WIDTH   = 24;
    parameter FRAC_BITS   = 5;
    parameter BETA        = 30;
    parameter THRESHOLD   = 32;

    reg         clk, rst_n;
    reg         input_wr_en;
    reg  [9:0]  input_wr_addr;
    reg  signed [IN_WIDTH-1:0] input_wr_data;
    reg         start;
    wire [3:0]  predicted_digit;
    wire [7:0]  max_spike_count;
    wire        done;

    // ─── 實例化 SNN Top ───
    snn_top #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_HIDDEN(NUM_HIDDEN),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .NUM_STEPS(NUM_STEPS),
        .W_WIDTH(W_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .MEM_WIDTH(MEM_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .BETA(BETA),
        .THRESHOLD(THRESHOLD),
        .WEIGHT_FILE1("fc1_weight.hex"),
        .WEIGHT_FILE2("fc2_weight.hex")
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .input_wr_en(input_wr_en),
        .input_wr_addr(input_wr_addr),
        .input_wr_data(input_wr_data),
        .start(start),
        .predicted_digit(predicted_digit),
        .max_spike_count(max_spike_count),
        .done(done)
    );

    // ─── Clock (100MHz) ───
    initial clk = 0;
    always #5 clk = ~clk;

    // ─── 載入測試圖片 ───
    // 用 reg 陣列暫存，然後一筆一筆寫進 BRAM
    reg [7:0] test_image [0:783];

    // ─── VCD ───
    initial begin
        $dumpfile("tb_snn_top.vcd");
        $dumpvars(0, tb_snn_top);
    end

    // ─── 主測試 ───
    integer i, timeout;
    integer nonzero_count;

    initial begin
        $display("==================================================");
        $display("  SNN Top Module Testbench");
        $display("==================================================");
        $display("  Architecture: %0d -> %0d -> %0d",
                 NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS);
        $display("  Time steps: %0d", NUM_STEPS);
        $display("");

        // ─── Reset ───
        rst_n = 0;
        start = 0;
        input_wr_en = 0;
        input_wr_addr = 0;
        input_wr_data = 0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ─── 載入 hex 檔到暫存陣列 ───
        $display("--- Loading test image (input_000.hex) ---");
        $readmemh("input_000.hex", test_image);

        // 檢查有多少非零值（確認載入成功）
        nonzero_count = 0;
        for (i = 0; i < 784; i = i + 1) begin
            if (test_image[i] !== 8'hxx && test_image[i] != 0)
                nonzero_count = nonzero_count + 1;
        end
        $display("  Non-zero pixels: %0d / 784", nonzero_count);

        if (nonzero_count == 0) begin
            $display("  ERROR: No valid pixels loaded! Check input_000.hex");
            $display("  Make sure you ran: python fix_hex_files.py");
            $finish;
        end

        // ─── 寫入 input BRAM（每筆穩定 hold 一個 cycle）───
        $display("  Writing to input BRAM...");
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            @(posedge clk);
            input_wr_en   <= 1;
            input_wr_addr <= i;
            input_wr_data <= test_image[i];
        end
        @(posedge clk);
        input_wr_en <= 0;

        // 等幾個 cycle 確保最後一筆寫入完成
        repeat (4) @(posedge clk);

        // 驗證 BRAM 內容（抽查幾個位置）
        $display("  BRAM spot check:");
        $display("    bram[0]   = %0d", uut.input_bram[0]);
        $display("    bram[100] = %0d", uut.input_bram[100]);
        $display("    bram[400] = %0d", uut.input_bram[400]);
        $display("    bram[783] = %0d", uut.input_bram[783]);

        // ─── 啟動推論 ───
        $display("");
        $display("--- Starting inference (%0d time steps) ---", NUM_STEPS);
        $display("  (This may take 1-5 minutes in simulation...)");

        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // 等待完成
        timeout = 0;
        while (!done && timeout < 10_000_000) begin
            @(posedge clk);
            timeout = timeout + 1;
            if (timeout % 1_000_000 == 0)
                $display("  ... %0d M cycles elapsed", timeout / 1_000_000);
        end

        if (timeout >= 10_000_000) begin
            $display("  ERROR: Timeout after 10M cycles!");
            $finish;
        end

        @(posedge clk);
        #1;

        // ─── 結果 ───
        $display("");
        $display("==================================================");
        $display("  RESULTS");
        $display("==================================================");
        $display("  Predicted digit : %0d", predicted_digit);
        $display("  Max spike count : %0d", max_spike_count);
        $display("  Total cycles    : %0d", timeout);
        $display("  Simulated time  : %0d us at 100MHz", timeout / 100);
        $display("");

        $display("  Spike counts per output neuron:");
        $write("  [ ");
        for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
            $write("%3d", uut.spike_counts[i]);
            if (i < NUM_OUTPUTS - 1) $write(",");
        end
        $display(" ]");
        $display("");

        $display("  Compare with fpga_export/test_vectors/result_000.txt");
        $display("==================================================");

        $finish;
    end

endmodule
