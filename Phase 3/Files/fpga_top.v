// =============================================================
// fpga_top.v — Basys 3 FPGA 頂層模組
// =============================================================
//
// 連接 SNN 推論引擎到 Basys 3 的 I/O：
//   - USB-UART：PC 傳圖片進來，FPGA 傳結果回去
//   - 7-Segment Display：顯示預測的數字
//   - LED：顯示狀態
//
// 操作流程：
//   1. PC 透過 UART 送 784 個 bytes（一張 MNIST 圖片）
//   2. FPGA 自動開始推論
//   3. 推論完成後，FPGA 透過 UART 回傳預測結果
//   4. 7-segment 顯示預測的數字
//   5. LED[0] = 推論中（閃爍）, LED[15] = 推論完成
//
// =============================================================

module fpga_top (
    input  wire        clk,          // 100MHz Basys 3 clock
    input  wire        btnC,         // Center button = reset
    input  wire        RsRx,         // UART RX (from PC)
    output wire        RsTx,         // UART TX (to PC)
    output wire [6:0]  seg,          // 7-segment segments (active low)
    output wire [3:0]  an,           // 7-segment anodes (active low)
    output wire [15:0] led           // LEDs
);

    // ─── Reset（btnC = active high，轉成 active low）───
    wire rst_n = ~btnC;

    // ─── 參數 ───
    localparam NUM_INPUTS  = 784;
    localparam NUM_HIDDEN  = 128;
    localparam NUM_OUTPUTS = 10;
    localparam NUM_STEPS   = 25;

    // =========================================================
    // UART RX
    // =========================================================
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_rx (
        .clk(clk),
        .rst_n(rst_n),
        .rx(RsRx),
        .data_out(rx_data),
        .data_valid(rx_valid)
    );

    // =========================================================
    // UART TX
    // =========================================================
    reg  [7:0] tx_data;
    reg        tx_send;
    wire       tx_busy;

    uart_tx #(
        .CLK_FREQ(100_000_000),
        .BAUD_RATE(115200)
    ) u_tx (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tx_data),
        .send(tx_send),
        .tx(RsTx),
        .busy(tx_busy)
    );

    // =========================================================
    // SNN 推論引擎
    // =========================================================
    reg         snn_input_wr_en;
    reg  [9:0]  snn_input_wr_addr;
    reg  signed [7:0] snn_input_wr_data;
    reg         snn_start;
    wire [3:0]  snn_predicted_digit;
    wire [7:0]  snn_max_spike_count;
    wire        snn_done;

    snn_top #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_HIDDEN(NUM_HIDDEN),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .NUM_STEPS(NUM_STEPS),
        .WEIGHT_FILE1("fc1_weight.hex"),
        .WEIGHT_FILE2("fc2_weight.hex")
    ) u_snn (
        .clk(clk),
        .rst_n(rst_n),
        .input_wr_en(snn_input_wr_en),
        .input_wr_addr(snn_input_wr_addr),
        .input_wr_data(snn_input_wr_data),
        .start(snn_start),
        .predicted_digit(snn_predicted_digit),
        .max_spike_count(snn_max_spike_count),
        .done(snn_done)
    );

    // =========================================================
    // 主控 FSM：接收圖片 → 推論 → 回傳結果
    // =========================================================
    localparam F_IDLE     = 3'd0;   // 等待同步 byte (0xAA)
    localparam F_SYNC     = 3'd1;   // 收到 0xAA，準備接收
    localparam F_RECEIVE  = 3'd2;   // 接收 784 bytes
    localparam F_INFER    = 3'd3;   // 啟動推論
    localparam F_WAIT     = 3'd4;   // 等推論完成
    localparam F_SEND     = 3'd5;   // 回傳結果
    localparam F_DONE     = 3'd6;   // 完成

    reg [2:0]  fstate;
    reg [9:0]  pixel_count;
    reg [3:0]  result_digit;
    reg [7:0]  result_spike_count;
    reg [3:0]  send_idx;            // 傳送計數器（0=digit, 1~10=spike counts）

    // 讀取 snn_top 裡的 spike_counts 陣列
    wire [7:0] snn_spike_count_byte;
    assign snn_spike_count_byte = u_snn.spike_counts[send_idx - 1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fstate            <= F_IDLE;
            pixel_count       <= 0;
            snn_input_wr_en   <= 0;
            snn_input_wr_addr <= 0;
            snn_input_wr_data <= 0;
            snn_start         <= 0;
            tx_send           <= 0;
            tx_data           <= 0;
            result_digit      <= 0;
            result_spike_count <= 0;
            send_idx          <= 0;
        end
        else begin
            snn_input_wr_en <= 0;
            snn_start       <= 0;
            tx_send         <= 0;

            case (fstate)

                F_IDLE: begin
                    pixel_count <= 0;
                    // 等待同步 byte 0xAA
                    if (rx_valid) begin
                        if (rx_data == 8'hAA) begin
                            fstate <= F_SYNC;
                        end
                        // 不是 0xAA → 忽略（丟掉垃圾資料）
                    end
                end

                F_SYNC: begin
                    // 收到 0xAA，開始接收圖片資料
                    if (rx_valid) begin
                        snn_input_wr_en   <= 1;
                        snn_input_wr_addr <= 0;
                        snn_input_wr_data <= rx_data;
                        pixel_count       <= 1;
                        fstate            <= F_RECEIVE;
                    end
                end

                F_RECEIVE: begin
                    if (rx_valid) begin
                        snn_input_wr_en   <= 1;
                        snn_input_wr_addr <= pixel_count;
                        snn_input_wr_data <= rx_data;
                        pixel_count       <= pixel_count + 1;

                        if (pixel_count == NUM_INPUTS - 1) begin
                            fstate <= F_INFER;
                        end
                    end
                end

                F_INFER: begin
                    // 啟動推論
                    snn_start <= 1;
                    fstate    <= F_WAIT;
                end

                F_WAIT: begin
                    // 等推論完成
                    if (snn_done) begin
                        result_digit       <= snn_predicted_digit;
                        result_spike_count <= snn_max_spike_count;
                        send_idx           <= 0;
                        fstate             <= F_SEND;
                    end
                end

                F_SEND: begin
                    // 回傳 11 bytes: 1 byte 預測 + 10 bytes spike counts
                    if (!tx_busy && !tx_send) begin
                        if (send_idx == 0) begin
                            tx_data <= {4'd0, result_digit};
                            tx_send <= 1;
                            send_idx <= 1;
                        end
                        else if (send_idx <= NUM_OUTPUTS) begin
                            tx_data <= snn_spike_count_byte;
                            tx_send <= 1;
                            send_idx <= send_idx + 1;
                        end
                        else begin
                            fstate <= F_DONE;
                        end
                    end
                end

                F_DONE: begin
                    // 等 TX 完成，然後回 IDLE 準備接收下一張
                    if (!tx_busy && !tx_send) begin
                        fstate <= F_IDLE;
                    end
                end

            endcase
        end
    end

    // =========================================================
    // 7-Segment Display：顯示預測數字
    // =========================================================
    // Basys 3 的 7-segment 是 active-low，共 4 位
    // 我們只用最右邊那位（an[0]），顯示 0~9

    reg [6:0] seg_pattern;

    always @(*) begin
        case (result_digit)
            4'd0: seg_pattern = 7'b1000000;
            4'd1: seg_pattern = 7'b1111001;
            4'd2: seg_pattern = 7'b0100100;
            4'd3: seg_pattern = 7'b0110000;
            4'd4: seg_pattern = 7'b0011001;
            4'd5: seg_pattern = 7'b0010010;
            4'd6: seg_pattern = 7'b0000010;
            4'd7: seg_pattern = 7'b1111000;
            4'd8: seg_pattern = 7'b0000000;
            4'd9: seg_pattern = 7'b0010000;
            default: seg_pattern = 7'b0111111;  // dash
        endcase
    end

    assign seg = seg_pattern;    // seg_pattern 已經是 active-low，不用再取反
    assign an  = 4'b1110;        // 只啟用最右邊的 digit

    // =========================================================
    // LED 狀態指示
    // =========================================================
    // LED[0]  = 推論中（用慢速閃爍）
    // LED[15] = 推論完成
    // LED[3:1] = 目前 FSM 狀態

    reg [23:0] blink_counter;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) blink_counter <= 0;
        else        blink_counter <= blink_counter + 1;
    end

    assign led[0]    = (fstate == F_WAIT) & blink_counter[22];  // ~3Hz 閃爍
    assign led[3:1]  = fstate;
    assign led[15]   = (fstate == F_DONE);
    assign led[14:4] = 0;

endmodule
