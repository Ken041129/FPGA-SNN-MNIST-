// =============================================================
// mac.v — Multiply-Accumulate 單元
// =============================================================
//
// 功能：計算一個神經元的突觸電流
//   cur = Σ (W[j] × input[j])  for j = 0 to NUM_INPUTS-1
//   然後右移 FRAC_BITS 位對齊
//
// Day 4 Python 對應：
//   cur1 = np.dot(w1_q[i], input_q) >> FRAC_BITS
//
// 運作方式（時分多工）：
//   1. start=1 → 清空累加器，開始從 addr=0 讀 BRAM
//   2. 每個 clock cycle：
//      - 從 BRAM 讀 W[addr]
//      - 乘以 input[addr]
//      - 累加到 acc
//      - addr++
//   3. addr 到 NUM_INPUTS → 輸出 acc >> FRAC_BITS, done=1
//
// 介面：
//   start       — 脈衝啟動（1 cycle 高態）
//   weight_data — 來自 BRAM 的權重（8-bit signed）
//   weight_addr — 輸出給 BRAM 的讀取地址
//   input_data  — 輸入值（8-bit，從 input buffer 讀取）
//   input_addr  — 輸出給 input buffer 的讀取地址
//   result      — 累加結果（右移後）
//   done        — 完成信號（1 cycle 高態）
//
// =============================================================

module mac #(
    parameter NUM_INPUTS  = 784,     // 輸入數量（fc1=784, fc2=128）
    parameter W_WIDTH     = 8,       // 權重位寬
    parameter IN_WIDTH    = 8,       // 輸入位寬
    parameter ACC_WIDTH   = 24,      // 累加器位寬（要夠寬！）
    parameter OUT_WIDTH   = 16,      // 輸出位寬
    parameter FRAC_BITS   = 5,       // 定點數小數位數
    parameter ADDR_WIDTH  = 10       // 地址位寬 (2^10=1024 > 784)
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,       // 啟動信號

    // BRAM 介面（讀權重）
    output reg  [ADDR_WIDTH-1:0]      weight_addr, // 權重地址
    input  wire signed [W_WIDTH-1:0]  weight_data, // 權重資料

    // Input buffer 介面
    output wire [ADDR_WIDTH-1:0]      input_addr,  // 輸入地址
    input  wire signed [IN_WIDTH-1:0] input_data,  // 輸入資料

    // 輸出
    output reg  signed [OUT_WIDTH-1:0] result,     // 最終結果
    output reg                         done        // 完成旗標
);

    // ─── 狀態機 ───
    localparam IDLE    = 2'd0;
    localparam COMPUTE = 2'd1;
    localparam FINISH  = 2'd2;

    reg [1:0] state;

    // ─── 累加器 ───
    //   為什麼需要 24-bit？
    //   最壞情況：784 個 8-bit × 8-bit 乘積（最大 127×127=16129）累加
    //   784 × 16129 = 12,645,136 → 需要 24-bit（2^24 = 16,777,216）
    reg signed [ACC_WIDTH-1:0] acc;

    // ─── 乘法結果 ───
    wire signed [W_WIDTH+IN_WIDTH-1:0] product;
    assign product = weight_data * input_data;  // 8-bit × 8-bit = 16-bit

    // ─── 計數器 ───
    reg [ADDR_WIDTH-1:0] count;

    // ─── input_addr 跟 weight_addr 同步 ───
    assign input_addr = weight_addr;

    // ─── BRAM 有 1 cycle 讀取延遲 ───
    // 因此我們需要一個 pipeline：
    //   cycle 0: 送出 addr=0
    //   cycle 1: BRAM 回傳 data[0]，同時送出 addr=1
    //   cycle 2: BRAM 回傳 data[1]，累加 data[0] 的乘積
    //   ...
    // 用 valid 信號標記「BRAM 的資料已經準備好了」

    reg bram_valid;  // 延遲一個 cycle 的有效信號

    // ─── 狀態機 + 運算 ───
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            acc         <= 0;
            weight_addr <= 0;
            count       <= 0;
            bram_valid  <= 0;
            result      <= 0;
            done        <= 0;
        end
        else begin
            done <= 0;  // done 只維持一個 cycle

            case (state)

                IDLE: begin
                    if (start) begin
                        // 啟動：清空累加器，從地址 0 開始讀
                        acc         <= 0;
                        weight_addr <= 0;
                        count       <= 0;
                        bram_valid  <= 0;
                        state       <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // --- Pipeline Stage 1: 送地址 ---
                    if (count < NUM_INPUTS) begin
                        weight_addr <= count + 1; // 預取下一個地址
                        count       <= count + 1;
                    end

                    // --- Pipeline Stage 2: 累加（上一個 cycle 的資料）---
                    if (count > 0 && count <= NUM_INPUTS) begin
                        acc <= acc + product;
                    end

                    if (count == NUM_INPUTS) begin
                        state <= FINISH;
                    end
                end

                FINISH: begin
                    // 右移對齊 + 輸出結果
                    // Python: cur = cur >> FRAC_BITS
                    result <= acc >>> FRAC_BITS;
                    done   <= 1;
                    state  <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule
