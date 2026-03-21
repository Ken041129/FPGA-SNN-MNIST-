// =============================================================
// lif.v — Leaky Integrate-and-Fire 神經元模組
// =============================================================
//
// 這個模組實作 Day 4 的四行公式：
//   mem = (BETA * mem) >> FRAC_BITS   // 衰減
//   mem = mem + cur                    // 累加
//   spk = (mem >= THRESHOLD)           // 判斷
//   if (spk) mem = mem - THRESHOLD     // reset
//
// 介面：
//   clk, rst_n    — 時脈和非同步低態重置
//   en            — 致能（高態時更新膜電位）
//   cur_in        — 突觸電流（來自 MAC 單元）
//   spk_out       — spike 輸出（1-bit）
//   mem_out        — 膜電位輸出（用於 debug / testbench 觀察）
//
// 參數：
//   MEM_WIDTH     — 膜電位位寬（預設 16）
//   CUR_WIDTH     — 輸入電流位寬（預設 16）
//   FRAC_BITS     — 定點數小數位數（預設 5）
//   BETA          — 衰減係數的定點表示（預設 30 = 0.95 × 32）
//   THRESHOLD     — 閾值的定點表示（預設 32 = 1.0 × 32）
//
// =============================================================

module lif #(
    parameter MEM_WIDTH  = 16,       // 膜電位位寬（比權重寬，避免溢位）
    parameter CUR_WIDTH  = 16,       // 輸入電流位寬
    parameter FRAC_BITS  = 5,        // 小數位數
    parameter BETA       = 30,       // 0.95 × 2^5 = 30.4 ≈ 30
    parameter THRESHOLD  = 32        // 1.0 × 2^5 = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,    // active-low reset
    input  wire                     en,       // 致能信號
    input  wire signed [CUR_WIDTH-1:0] cur_in,   // 突觸電流
    output wire                     spk_out,  // spike 輸出
    output wire signed [MEM_WIDTH-1:0] mem_out   // 膜電位（debug 用）
);

    // ─── 內部信號 ───
    reg  signed [MEM_WIDTH-1:0] mem;        // 膜電位暫存器

    // 中間運算需要更寬的位數，避免乘法溢位
    // BETA (≤32, 6-bit) × mem (16-bit) = 最多 22-bit
    wire signed [MEM_WIDTH+6-1:0] decay_product;
    wire signed [MEM_WIDTH-1:0]   decayed_mem;
    wire signed [MEM_WIDTH-1:0]   new_mem;
    wire                          spike;

    // ─── Step 1: 膜電位衰減 ───
    // Python: mem = (beta * mem) >> FRAC_BITS
    //
    // 定點數乘法：BETA 是 Q2.5，mem 是 Q?.5
    // 乘積是 Q?.10，右移 5 位變回 Q?.5
    // 用算術右移（>>>）保留符號位
    assign decay_product = BETA * mem;
    assign decayed_mem   = decay_product >>> FRAC_BITS;

    // ─── Step 2: 累加輸入電流 ───
    // Python: mem = mem + cur
    assign new_mem = decayed_mem + cur_in;

    // ─── Step 3: Spike 判斷 ───
    // Python: spk = (mem >= threshold)
    assign spike = (new_mem >= THRESHOLD);

    // ─── 輸出 ───
    assign spk_out = spike;
    assign mem_out = mem;

    // ─── 時序邏輯：更新膜電位 ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset：膜電位歸零
            mem <= 0;
        end
        else if (en) begin
            // Step 4: Reset by subtraction
            // Python: if (spk) mem = mem - threshold
            if (spike)
                mem <= new_mem - THRESHOLD;
            else
                mem <= new_mem;
        end
        // else: en=0 時，膜電位保持不變
    end

endmodule
