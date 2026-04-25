`timescale 1ns / 1ps

// =============================================================================
// Firing Rate Decoder — sliding-window spike counter → seizure alert
// =============================================================================
// • Counts layer-2 spikes over a sliding window
// • seizure_detected = (running_count >= FIRE_THRESHOLD)
// • Defaults tuned LOW for guaranteed visibility
// =============================================================================

module firing_rate_decoder #(
    parameter integer WINDOW_SIZE      = 4096,    // sliding window for spike counting
    parameter integer POST_ICTAL_BLANK = 256,     // post-ictal blanking period
    parameter integer FIRE_THRESHOLD   = 600,     // spike count threshold for seizurely
    parameter integer RATE_BITS        = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 enable,
    input  wire                 spike_in,
    output reg  [RATE_BITS-1:0] firing_rate,
    output reg                  seizure_detected,
    output reg                  classification_valid
);

    function integer clog2;
        input integer value;
        integer tmp;
        begin
            tmp = value - 1;
            clog2 = 0;
            while (tmp > 0) begin
                tmp = tmp >> 1;
                clog2 = clog2 + 1;
            end
        end
    endfunction

    localparam integer PTR_BITS   = clog2(WINDOW_SIZE);
    localparam integer COUNT_BITS = PTR_BITS + 1;
    localparam integer BLANK_BITS = (POST_ICTAL_BLANK > 0) ? clog2(POST_ICTAL_BLANK + 1) : 1;

    reg [WINDOW_SIZE-1:0]  spike_history;
    reg [PTR_BITS-1:0]     write_ptr;
    reg [COUNT_BITS-1:0]   running_count;
    reg [COUNT_BITS-1:0]   samples_seen;
    reg [BLANK_BITS-1:0]   blanking_count;

    reg                    old_spike;
    reg [COUNT_BITS-1:0]   next_count;

    function [RATE_BITS-1:0] count_to_rate;
        input [COUNT_BITS-1:0] cnt;
        reg [31:0] padded_cnt; 
        reg [31:0] max_val;
        begin
            // Zero-extend the count to a safe 32-bit width
            padded_cnt = cnt; 
            
            // Calculate the maximum value representable by RATE_BITS
            max_val = (1 << RATE_BITS) - 1;
            
            // Saturate if the count exceeds the max value
            if (padded_cnt > max_val) begin
                count_to_rate = {RATE_BITS{1'b1}};
            end else begin
                // Safe to slice because padded_cnt is explicitly 32 bits
                count_to_rate = padded_cnt[RATE_BITS-1:0];
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_history        <= {WINDOW_SIZE{1'b0}};
            write_ptr            <= {PTR_BITS{1'b0}};
            running_count        <= {COUNT_BITS{1'b0}};
            samples_seen         <= {COUNT_BITS{1'b0}};
            blanking_count       <= {BLANK_BITS{1'b0}};
            firing_rate          <= {RATE_BITS{1'b0}};
            seizure_detected     <= 1'b0;
            classification_valid <= 1'b0;
        end else begin
            seizure_detected     <= 1'b0;
            classification_valid <= 1'b0;

            if (enable) begin
                old_spike = spike_history[write_ptr];
                spike_history[write_ptr] <= spike_in;

                if (write_ptr == WINDOW_SIZE - 1)
                    write_ptr <= {PTR_BITS{1'b0}};
                else
                    write_ptr <= write_ptr + 1'b1;

                // Update running count
                next_count = running_count;
                if (spike_in && !old_spike)
                    next_count = running_count + 1'b1;
                else if (!spike_in && old_spike)
                    next_count = (running_count > 0) ? running_count - 1'b1 : {COUNT_BITS{1'b0}};
                running_count <= next_count;

                if (samples_seen < WINDOW_SIZE)
                    samples_seen <= samples_seen + 1'b1;

                firing_rate <= count_to_rate(next_count);

                // Classification valid once we have a full window
                if (samples_seen + 1'b1 >= WINDOW_SIZE) begin
                    classification_valid <= 1'b1;

                    if (blanking_count != {BLANK_BITS{1'b0}}) begin
                        blanking_count <= blanking_count - 1'b1;
                    end else if (next_count >= FIRE_THRESHOLD) begin
                        seizure_detected <= 1'b1;
                        if (POST_ICTAL_BLANK > 0)
                            blanking_count <= POST_ICTAL_BLANK[BLANK_BITS-1:0];
                    end
                end
            end
        end
    end

endmodule
