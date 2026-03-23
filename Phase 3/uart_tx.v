// =============================================================
// uart_tx.v — UART 傳送器
// =============================================================
//
// 傳送 8-bit 資料回 PC
// 協議：8N1, 115200 baud
//
// =============================================================

module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data_in,     // 要傳送的 byte
    input  wire       send,        // 高態一個 cycle = 開始傳送
    output reg        tx,          // UART TX pin
    output wire       busy         // 正在傳送中
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_count;
    reg [2:0]  bit_idx;
    reg [7:0]  tx_shift;

    assign busy = (state != IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            clk_count <= 0;
            bit_idx   <= 0;
            tx_shift  <= 0;
            tx        <= 1;  // UART idle = high
        end
        else begin
            case (state)

                IDLE: begin
                    tx <= 1;
                    if (send) begin
                        tx_shift  <= data_in;
                        clk_count <= 0;
                        state     <= START;
                    end
                end

                START: begin
                    tx <= 0;  // Start bit = low
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        bit_idx   <= 0;
                        state     <= DATA;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DATA: begin
                    tx <= tx_shift[bit_idx];  // LSB first
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        if (bit_idx == 7)
                            state <= STOP;
                        else
                            bit_idx <= bit_idx + 1;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                STOP: begin
                    tx <= 1;  // Stop bit = high
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        state     <= IDLE;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

            endcase
        end
    end

endmodule
