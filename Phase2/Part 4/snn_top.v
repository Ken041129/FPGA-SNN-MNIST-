// =============================================================
// snn_top.v — SNN 推論引擎 Top Module
// =============================================================
//
// 完整的兩層 SNN 推論：
//   1. 外部把 784 個像素寫入 input BRAM
//   2. start=1 啟動推論
//   3. 跑 NUM_STEPS 個 time step：
//      每步：Layer1(784→128) → Layer2(128→10)
//   4. 統計 10 個輸出神經元的 spike 次數
//   5. 輸出 spike 最多的神經元 = 預測的數字
//
// Day 4 Python 對應：
//   for t in range(NUM_STEPS):
//       cur1 = W1 @ input >> 5
//       mem1 = beta*mem1>>5 + cur1; spk1 = ...
//       cur2 = W2 @ spk1
//       mem2 = beta*mem2>>5 + cur2; spk2 = ...
//   predicted = argmax(sum(spk2_record))
//
// =============================================================

module snn_top #(
    parameter NUM_INPUTS   = 784,
    parameter NUM_HIDDEN   = 128,
    parameter NUM_OUTPUTS  = 10,
    parameter NUM_STEPS    = 25,
    parameter W_WIDTH      = 8,
    parameter IN_WIDTH     = 8,
    parameter MEM_WIDTH    = 16,
    parameter ACC_WIDTH    = 24,
    parameter FRAC_BITS    = 5,
    parameter BETA         = 30,
    parameter THRESHOLD    = 15,
    parameter WEIGHT_FILE1 = "fc1_weight.hex",
    parameter WEIGHT_FILE2 = "fc2_weight.hex"
)(
    input  wire        clk,
    input  wire        rst_n,

    // ─── 輸入介面：外部寫入圖片 ───
    input  wire        input_wr_en,           // 寫入致能
    input  wire [9:0]  input_wr_addr,         // 寫入地址 (0~783)
    input  wire signed [IN_WIDTH-1:0] input_wr_data,  // 寫入資料

    // ─── 控制 ───
    input  wire        start,                 // 啟動推論

    // ─── 輸出 ───
    output reg  [3:0]  predicted_digit,       // 預測結果 (0~9)
    output reg  [7:0]  max_spike_count,       // 最高 spike 數
    output reg         done                   // 推論完成
);

    // =========================================================
    // Input BRAM：存放 784 個像素
    // =========================================================
    reg signed [IN_WIDTH-1:0] input_bram [0:NUM_INPUTS-1];

    // 外部寫入
    always @(posedge clk) begin
        if (input_wr_en)
            input_bram[input_wr_addr] <= input_wr_data;
    end

    // Layer 1 讀取（1-cycle 延遲）
    wire [9:0] l1_input_addr;
    reg  signed [IN_WIDTH-1:0] l1_input_data;

    always @(posedge clk) begin
        l1_input_data <= input_bram[l1_input_addr];
    end

    // =========================================================
    // Layer 1: 784 → 128
    // =========================================================
    reg  l1_start;
    wire [NUM_HIDDEN-1:0] l1_spk_out;
    wire l1_done;

    fc_layer #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(NUM_HIDDEN),
        .W_WIDTH(W_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .MEM_WIDTH(MEM_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .BETA(BETA),
        .THRESHOLD(THRESHOLD),
        .WEIGHT_FILE(WEIGHT_FILE1),
        .ADDR_WIDTH(10),
        .MAC_FRAC_BITS(FRAC_BITS)  // 輸入是 Q2.5 定點數，需要右移
    ) u_layer1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(l1_start),
        .input_addr(l1_input_addr),
        .input_data(l1_input_data),
        .spk_out(l1_spk_out),
        .done(l1_done)
    );

    // =========================================================
    // Spike Buffer：Layer 1 的 spike → Layer 2 的 input
    // =========================================================
    // Layer 1 輸出 128-bit spike vector
    // Layer 2 需要逐一讀取（像讀 BRAM 一樣）
    reg [NUM_HIDDEN-1:0] spk1_buffer;

    // Layer 2 讀取 spike buffer（1-cycle 延遲模擬）
    wire [9:0] l2_input_addr;
    reg  signed [IN_WIDTH-1:0] l2_input_data;

    always @(posedge clk) begin
        // spike 是 0 或 1，直接轉成 IN_WIDTH 位寬
        l2_input_data <= {{(IN_WIDTH-1){1'b0}}, spk1_buffer[l2_input_addr]};
    end

    // =========================================================
    // Layer 2: 128 → 10
    // =========================================================
    reg  l2_start;
    wire [NUM_OUTPUTS-1:0] l2_spk_out;
    wire l2_done;

    fc_layer #(
        .NUM_INPUTS(NUM_HIDDEN),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .W_WIDTH(W_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .MEM_WIDTH(MEM_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .BETA(BETA),
        .THRESHOLD(THRESHOLD),
        .WEIGHT_FILE(WEIGHT_FILE2),
        .ADDR_WIDTH(8),
        .MAC_FRAC_BITS(0)    // spike 是整數 0/1，不需要右移
    ) u_layer2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(l2_start),
        .input_addr(l2_input_addr),
        .input_data(l2_input_data),
        .spk_out(l2_spk_out),
        .done(l2_done)
    );

    // =========================================================
    // Spike Counters：統計每個輸出神經元的 spike 次數
    // =========================================================
    reg [7:0] spike_counts [0:NUM_OUTPUTS-1];

    // =========================================================
    // Top-level FSM
    // =========================================================
    localparam S_IDLE    = 3'd0;
    localparam S_L1_RUN  = 3'd1;    // Layer 1 計算中
    localparam S_L1_DONE = 3'd2;    // Layer 1 完成，暫存 spike
    localparam S_L2_RUN  = 3'd3;    // Layer 2 計算中
    localparam S_L2_DONE = 3'd4;    // Layer 2 完成，累加 spike count
    localparam S_OUTPUT  = 3'd5;    // 計算 argmax，輸出結果

    reg [2:0] state;
    reg [4:0] step_count;   // 0 ~ NUM_STEPS-1

    integer i;

    // ─── Argmax 邏輯 ───
    reg [3:0]  best_idx;
    reg [7:0]  best_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            step_count     <= 0;
            l1_start       <= 0;
            l2_start       <= 0;
            spk1_buffer    <= 0;
            predicted_digit <= 0;
            max_spike_count <= 0;
            done           <= 0;

            for (i = 0; i < NUM_OUTPUTS; i = i + 1)
                spike_counts[i] <= 0;
        end
        else begin
            l1_start <= 0;
            l2_start <= 0;
            done     <= 0;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        step_count <= 0;

                        // 清空 spike counters
                        for (i = 0; i < NUM_OUTPUTS; i = i + 1)
                            spike_counts[i] <= 0;

                        // 啟動 Layer 1 的第一個 time step
                        l1_start <= 1;
                        state    <= S_L1_RUN;
                    end
                end

                S_L1_RUN: begin
                    // 等 Layer 1 完成
                    if (l1_done) begin
                        state <= S_L1_DONE;
                    end
                end

                S_L1_DONE: begin
                    // 暫存 Layer 1 的 spike 到 buffer
                    spk1_buffer <= l1_spk_out;

                    // 啟動 Layer 2
                    l2_start <= 1;
                    state    <= S_L2_RUN;
                end

                S_L2_RUN: begin
                    // 等 Layer 2 完成
                    if (l2_done) begin
                        state <= S_L2_DONE;
                    end
                end

                S_L2_DONE: begin
                    // 累加 spike counts
                    for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                        if (l2_spk_out[i])
                            spike_counts[i] <= spike_counts[i] + 1;
                    end

                    // 下一個 time step 或結束
                    if (step_count == NUM_STEPS - 1) begin
                        state <= S_OUTPUT;
                    end
                    else begin
                        step_count <= step_count + 1;
                        l1_start   <= 1;
                        state      <= S_L1_RUN;
                    end
                end

                S_OUTPUT: begin
                    // Argmax：找 spike 最多的神經元
                    best_idx   = 0;
                    best_count = spike_counts[0];

                    for (i = 1; i < NUM_OUTPUTS; i = i + 1) begin
                        if (spike_counts[i] > best_count) begin
                            best_count = spike_counts[i];
                            best_idx   = i[3:0];
                        end
                    end

                    predicted_digit <= best_idx;
                    max_spike_count <= best_count;
                    done            <= 1;
                    state           <= S_IDLE;
                end

            endcase
        end
    end

endmodule
