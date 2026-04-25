`timescale 1ns / 1ps

// =============================================================================
// EEG Amplitude Encoder — One-hot spike bus (64 bits)
// =============================================================================
// Each of the 4 EEG channels maps to a 16-bit sub-bus [15:0].
// The amplitude is quantised to a level 0..15 and ONLY that single bit is SET.
//
// This matches the paper's encoding: "A single spike is produced on one of
// the amplitude levels for each newly acquired sample" — one-hot, not
// thermometer. Total active spikes per sample = exactly 4 (one per channel).
//
// The encoder also has an absolute-value threshold: if |eeg_input| < ABS_THRESH,
// no spikes are generated for that channel (quiescent suppression).
// =============================================================================

module eeg_amplitude_encoder #(
    parameter integer SAMPLE_BITS = 16,
    parameter signed [SAMPLE_BITS-1:0] CH0_MIN = -16'sd32768,
    parameter signed [SAMPLE_BITS-1:0] CH0_MAX =  16'sd32767,
    parameter signed [SAMPLE_BITS-1:0] CH1_MIN = -16'sd32768,
    parameter signed [SAMPLE_BITS-1:0] CH1_MAX =  16'sd32767,
    parameter signed [SAMPLE_BITS-1:0] CH2_MIN = -16'sd32768,
    parameter signed [SAMPLE_BITS-1:0] CH2_MAX =  16'sd32767,
    parameter signed [SAMPLE_BITS-1:0] CH3_MIN = -16'sd32768,
    parameter signed [SAMPLE_BITS-1:0] CH3_MAX =  16'sd32767,
    parameter integer ABS_THRESH = 0    // quiescent threshold (0 = disabled)
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          sample_valid,
    input  wire signed [SAMPLE_BITS-1:0] eeg_ch0,
    input  wire signed [SAMPLE_BITS-1:0] eeg_ch1,
    input  wire signed [SAMPLE_BITS-1:0] eeg_ch2,
    input  wire signed [SAMPLE_BITS-1:0] eeg_ch3,
    output reg  [63:0]                   spike_bus,
    output reg                           spike_valid
);

    // -------------------------------------------------------------------------
    // Convert amplitude to a level 0..15 within [level_min, level_max]
    // -------------------------------------------------------------------------
    function [3:0] amplitude_to_level;
        input signed [SAMPLE_BITS-1:0] sample;
        input signed [SAMPLE_BITS-1:0] level_min;
        input signed [SAMPLE_BITS-1:0] level_max;
        integer range_ext, offset_ext, scaled;
        begin
            if (level_max <= level_min) begin
                amplitude_to_level = 4'd0;
            end else if (sample <= level_min) begin
                amplitude_to_level = 4'd0;
            end else if (sample >= level_max) begin
                amplitude_to_level = 4'd15;
            end else begin
                range_ext  = $signed(level_max) - $signed(level_min);
                offset_ext = $signed(sample) - $signed(level_min);
                scaled     = (offset_ext * 16) / (range_ext + 1);
                if (scaled < 0)       amplitude_to_level = 4'd0;
                else if (scaled > 15) amplitude_to_level = 4'd15;
                else                  amplitude_to_level = scaled[3:0];
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // One-hot mask: level → exactly 1 bit set at position [level]
    // e.g. level=0 → 16'h0001, level=5 → 16'h0020, level=15 → 16'h8000
    // -------------------------------------------------------------------------
    function [15:0] onehot_mask;
        input [3:0] level;
        begin
            onehot_mask = 16'd1 << level;
        end
    endfunction

    reg [3:0]  level0, level1, level2, level3;
    reg [15:0] mask0,  mask1,  mask2,  mask3;
    reg [63:0] spike_bus_next;

    // Absolute value helper
    function [SAMPLE_BITS-1:0] abs_val;
        input signed [SAMPLE_BITS-1:0] v;
        begin
            abs_val = (v < 0) ? (~v + 1'b1) : v;
        end
    endfunction

    always @* begin
        level0 = amplitude_to_level(eeg_ch0, CH0_MIN, CH0_MAX);
        level1 = amplitude_to_level(eeg_ch1, CH1_MIN, CH1_MAX);
        level2 = amplitude_to_level(eeg_ch2, CH2_MIN, CH2_MAX);
        level3 = amplitude_to_level(eeg_ch3, CH3_MIN, CH3_MAX);

        // One-hot masks (exactly 1 bit per channel)
        mask0 = onehot_mask(level0);
        mask1 = onehot_mask(level1);
        mask2 = onehot_mask(level2);
        mask3 = onehot_mask(level3);

        // Apply quiescent suppression
        if (ABS_THRESH > 0 && abs_val(eeg_ch0) < ABS_THRESH) mask0 = 16'd0;
        if (ABS_THRESH > 0 && abs_val(eeg_ch1) < ABS_THRESH) mask1 = 16'd0;
        if (ABS_THRESH > 0 && abs_val(eeg_ch2) < ABS_THRESH) mask2 = 16'd0;
        if (ABS_THRESH > 0 && abs_val(eeg_ch3) < ABS_THRESH) mask3 = 16'd0;

        spike_bus_next = {mask3, mask2, mask1, mask0};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_bus   <= 64'd0;
            spike_valid <= 1'b0;
        end else if (sample_valid) begin
            spike_bus   <= spike_bus_next;
            spike_valid <= 1'b1;
        end else begin
            spike_valid <= 1'b0;
        end
    end

endmodule
