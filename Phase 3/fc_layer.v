// =============================================================
// fc_layer.v — 全連接層模組
// =============================================================
//
// 功能：完成一整層的 SNN 推論（一個 time step）
//   對每個輸出神經元 i (0 ~ NUM_OUTPUTS-1)：
//     1. MAC: cur[i] = (Σ W[i][j] × input[j]) >> FRAC_BITS
//     2. LIF: mem[i] 衰減 → 加上 cur[i] → 判斷 spike → reset
//
// Day 4 Python 對應：
//   for i in range(NUM_OUTPUTS):
//       cur[i] = np.dot(w[i], input) >> FRAC_BITS
//       mem[i] = (beta * mem[i]) >> FRAC_BITS + cur[i]
//       spk[i] = (mem[i] >= threshold)
//       mem[i] -= spk[i] * threshold
//
// 架構：
//   - 1 個 MAC 單元（時分多工，依序服務每個神經元）
//   - 1 個 LIF 單元（同樣時分多工）
//   - 1 塊 Weight BRAM（存所有權重）
//   - 1 組 membrane state 陣列（每個神經元的膜電位）
//   - 1 組 spike 暫存器（紀錄所有輸出 spike）
//
// 介面：
//   start       — 啟動一個 time step 的推論
//   input_data  — 從外部讀入的輸入資料
//   input_addr  — 輸出給外部 input buffer 的地址
//   spk_out     — 所有輸出神經元的 spike（NUM_OUTPUTS bits）
//   done        — 這個 time step 完成
//
// =============================================================

module fc_layer #(
    parameter NUM_INPUTS   = 784,
    parameter NUM_OUTPUTS  = 128,
    parameter W_WIDTH      = 8,
    parameter IN_WIDTH     = 8,
    parameter MEM_WIDTH    = 16,
    parameter ACC_WIDTH    = 24,
    parameter FRAC_BITS    = 5,
    parameter BETA         = 30,
    parameter THRESHOLD    = 32,
    parameter WEIGHT_FILE  = "fc1_weight.hex",  // 權重 .hex 檔案路徑
    parameter ADDR_WIDTH   = 10,  // 要能定址 NUM_INPUTS
    parameter MAC_FRAC_BITS = 5   // MAC 輸出右移位數
                                  // Layer1: =FRAC_BITS (input 是 Q2.5 定點數)
                                  // Layer2: =0 (input 是 spike 0/1，不需要右移)
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      start,
    input  wire                      mem_clear,   // 清除膜電位（新圖片推論前）

    // 外部 input buffer 介面
    output wire [ADDR_WIDTH-1:0]     input_addr,
    input  wire signed [IN_WIDTH-1:0] input_data,

    // 輸出
    output reg  [NUM_OUTPUTS-1:0]    spk_out,   // 所有神經元的 spike
    output reg                       done
);

    // ─── Weight BRAM ───
    // 所有神經元的權重攤平存放：
    //   addr 0 ~ NUM_INPUTS-1           : neuron 0 的權重
    //   addr NUM_INPUTS ~ 2*NUM_INPUTS-1: neuron 1 的權重
    //   ...
    localparam WEIGHT_DEPTH = NUM_INPUTS * NUM_OUTPUTS;
    localparam W_ADDR_WIDTH = $clog2(WEIGHT_DEPTH);

    reg signed [W_WIDTH-1:0] weight_mem [0:WEIGHT_DEPTH-1];

    initial begin
        $readmemh(WEIGHT_FILE, weight_mem);
    end

    // BRAM 讀取（1-cycle 延遲）
    reg signed [W_WIDTH-1:0] weight_data_reg;
    wire [W_ADDR_WIDTH-1:0]  weight_read_addr;

    always @(posedge clk) begin
        weight_data_reg <= weight_mem[weight_read_addr];
    end

    // ─── Membrane State 陣列 ───
    // 每個神經元有自己的膜電位（跨 time step 保留）
    reg signed [MEM_WIDTH-1:0] mem_state [0:NUM_OUTPUTS-1];

    integer k;
    // Reset 所有膜電位
    // （注意：這在合成時會被 initial block 處理，
    //   實際 FPGA 上要靠 rst_n 信號）

    // ─── MAC 介面信號 ───
    reg                         mac_start;
    wire [ADDR_WIDTH-1:0]       mac_weight_addr;  // MAC 要的「local」地址 (0~NUM_INPUTS-1)
    wire [ADDR_WIDTH-1:0]       mac_input_addr;
    wire signed [MEM_WIDTH-1:0] mac_result;
    wire                        mac_done;

    // MAC 模組 — 你自己改的那個版本！
    mac #(
        .NUM_INPUTS(NUM_INPUTS),
        .W_WIDTH(W_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(MEM_WIDTH),
        .FRAC_BITS(MAC_FRAC_BITS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .start(mac_start),
        .weight_addr(mac_weight_addr),
        .weight_data(weight_data_reg),
        .input_addr(mac_input_addr),
        .input_data(input_data),
        .result(mac_result),
        .done(mac_done)
    );

    // MAC 的 weight_addr 是 0~NUM_INPUTS-1（local）
    // 用 base_addr 累加取代 neuron_idx × NUM_INPUTS 的乘法（修 timing violation）
    reg [$clog2(NUM_OUTPUTS)-1:0] neuron_idx;
    reg [W_ADDR_WIDTH-1:0] weight_base_addr;  // 每個神經元的起始地址
    assign weight_read_addr = weight_base_addr + mac_weight_addr;

    // input_addr 直接透傳給外部
    assign input_addr = mac_input_addr;

    // ─── LIF 運算 ───
    // 用暫存器打斷組合邏輯路徑（修 timing violation）
    reg signed [MEM_WIDTH-1:0] cur_mem_state;   // 從 mem_state 讀出的值（registered）
    reg signed [MEM_WIDTH-1:0] latched_mac_result;  // 鎖住 MAC 結果
    wire signed [MEM_WIDTH+6-1:0] decay_product;
    wire signed [MEM_WIDTH-1:0]   decayed_mem;
    wire signed [MEM_WIDTH-1:0]   new_mem;
    wire                          spike;

    assign decay_product = BETA * cur_mem_state;
    assign decayed_mem   = decay_product >>> FRAC_BITS;
    assign new_mem       = decayed_mem + latched_mac_result;  // 用鎖住的值
    assign spike         = (new_mem >= THRESHOLD);

    // ─── FSM ───
    localparam S_IDLE       = 3'd0;
    localparam S_MAC_RUN    = 3'd1;
    localparam S_LIF_PREP   = 3'd2;   // 讀取 mem_state（1 cycle pipeline）
    localparam S_LIF_UPDATE = 3'd3;
    localparam S_DONE       = 3'd4;

    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            neuron_idx      <= 0;
            weight_base_addr <= 0;
            cur_mem_state   <= 0;
            latched_mac_result <= 0;
            mac_start       <= 0;
            spk_out         <= 0;
            done            <= 0;

            // Reset 所有膜電位
            for (k = 0; k < NUM_OUTPUTS; k = k + 1)
                mem_state[k] <= 0;
        end
        else begin
            mac_start <= 0;
            done      <= 0;

            // 清除膜電位（在開始新推論前由 snn_top 觸發）
            if (mem_clear) begin
                for (k = 0; k < NUM_OUTPUTS; k = k + 1)
                    mem_state[k] <= 0;
                cur_mem_state      <= 0;
                latched_mac_result <= 0;
            end

            case (state)

                S_IDLE: begin
                    if (start) begin
                        neuron_idx       <= 0;
                        weight_base_addr <= 0;
                        spk_out          <= 0;
                        mac_start        <= 1;
                        state            <= S_MAC_RUN;
                    end
                end

                S_MAC_RUN: begin
                    if (mac_done) begin
                        // MAC 完成：立刻鎖住結果 + 讀出膜電位
                        latched_mac_result <= mac_result;
                        cur_mem_state      <= mem_state[neuron_idx];
                        state              <= S_LIF_PREP;
                    end
                end

                S_LIF_PREP: begin
                    // cur_mem_state 已穩定，LIF 組合邏輯可以正確計算
                    state <= S_LIF_UPDATE;
                end

                S_LIF_UPDATE: begin
                    // LIF 更新膜電位
                    if (spike) begin
                        mem_state[neuron_idx] <= new_mem - THRESHOLD;
                        spk_out[neuron_idx]   <= 1'b1;
                    end
                    else begin
                        mem_state[neuron_idx] <= new_mem;
                    end

                    // 下一個神經元，或全部做完
                    if (neuron_idx == NUM_OUTPUTS - 1) begin
                        state <= S_DONE;
                    end
                    else begin
                        neuron_idx       <= neuron_idx + 1;
                        weight_base_addr <= weight_base_addr + NUM_INPUTS;
                        mac_start        <= 1;
                        state            <= S_MAC_RUN;
                    end
                end

                S_DONE: begin
                    done  <= 1;
                    state <= S_IDLE;
                end

            endcase
        end
    end

endmodule
