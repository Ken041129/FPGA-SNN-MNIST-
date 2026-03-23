// =============================================================
// uart_rx.v — UART 接收器
// =============================================================
//
// 從 PC 的 serial port 接收 8-bit 資料
// Basys 3 的 USB-UART 轉接晶片連接到 FPGA pin
//
// 協議：8N1（8 data bits, no parity, 1 stop bit）
// 預設 baud rate: 115200
//
// =============================================================

module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,  // Basys 3 = 100MHz
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,          // UART RX pin
    output reg  [7:0] data_out,    // 接收到的 byte
    output reg        data_valid   // 高態一個 cycle = 收到一個 byte
);

    // 每個 bit 需要的 clock cycle 數
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;  // 100M/115200 ≈ 868
    localparam HALF_BIT     = CLKS_PER_BIT / 2;      // 取樣在 bit 中間

    // 狀態
    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_count;   // bit 內的 cycle 計數
    reg [2:0]  bit_idx;     // 目前在第幾個 data bit (0~7)
    reg [7:0]  rx_shift;    // 移位暫存器

    // 輸入同步（防 metastability）
    reg rx_sync1, rx_sync2;
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            clk_count  <= 0;
            bit_idx    <= 0;
            rx_shift   <= 0;
            data_out   <= 0;
            data_valid <= 0;
        end
        else begin
            data_valid <= 0;

            case (state)

                IDLE: begin
                    // 等待 start bit（RX 從 1 變 0）
                    if (rx_sync2 == 0) begin
                        clk_count <= 0;
                        state     <= START;
                    end
                end

                START: begin
                    // 等到 start bit 中間確認還是 0
                    if (clk_count == HALF_BIT) begin
                        if (rx_sync2 == 0) begin
                            clk_count <= 0;
                            bit_idx   <= 0;
                            state     <= DATA;
                        end
                        else begin
                            // 假的 start bit，回 IDLE
                            state <= IDLE;
                        end
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DATA: begin
                    // 每 CLKS_PER_BIT 個 cycle 取樣一次
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        rx_shift[bit_idx] <= rx_sync2;  // LSB first

                        if (bit_idx == 7) begin
                            state <= STOP;
                        end
                        else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

                STOP: begin
                    // 等 stop bit 結束
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        data_out   <= rx_shift;
                        data_valid <= 1;
                        state      <= IDLE;
                        clk_count  <= 0;
                    end
                    else begin
                        clk_count <= clk_count + 1;
                    end
                end

            endcase
        end
    end

endmodule
