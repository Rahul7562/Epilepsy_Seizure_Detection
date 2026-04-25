`timescale 1ns / 1ps

// =============================================================================
// SNN Seizure Detection — Top Level
// =============================================================================
// Pipeline FSM: IDLE → ENCODE → LAYER1 → LAYER2 → DECODE → IDLE
// =============================================================================

module snn_seizure_top #(
    parameter L1_WEIGHT_FILE       = "layer1_weights.mem",
    parameter L2_WEIGHT_FILE       = "layer2_weights.mem",

    // ----- Neuron parameters -----
    // With one-hot encoding (4 active spikes/sample), typical L1 weighted
    // sum per timestep = 50-250 depending on amplitude level.
    // Threshold must be high enough to prevent constant firing from N0/N1.
    parameter integer L1_THRESHOLD = 200,
    parameter integer L1_LEAK      = 1,
    parameter integer L1_DECAY_SHIFT = 4,    // α = 15/16 = 0.9375

    parameter integer L2_THRESHOLD = 100,
    parameter integer L2_LEAK      = 0,
    parameter integer L2_DECAY_SHIFT = 4,    // α = 15/16 = 0.9375

    // ----- Encoder range (full 16-bit by default) -----
    parameter signed [15:0] ENC_CH0_MIN = -16'sd32768,
    parameter signed [15:0] ENC_CH0_MAX =  16'sd32767,
    parameter signed [15:0] ENC_CH1_MIN = -16'sd32768,
    parameter signed [15:0] ENC_CH1_MAX =  16'sd32767,
    parameter signed [15:0] ENC_CH2_MIN = -16'sd32768,
    parameter signed [15:0] ENC_CH2_MAX =  16'sd32767,
    parameter signed [15:0] ENC_CH3_MIN = -16'sd32768,
    parameter signed [15:0] ENC_CH3_MAX =  16'sd32767,

    // ----- Decoder parameters -----
    // Paper: window=4096, seizure target rate=0.35 (1434 spikes), 
    //        normal target rate=0.03 (123 spikes).
    // Threshold at ~600 to split the two distributions.
    parameter integer DECODER_WINDOW    = 4096,
    parameter integer DECODER_BLANKING  = 256,
    parameter integer DECODER_THRESHOLD = 1500
)(
    input  wire        clk_50m,
    input  wire        rst_n,
    input  wire        sample_valid,
    input  wire signed [15:0] eeg_ch0,
    input  wire signed [15:0] eeg_ch1,
    input  wire signed [15:0] eeg_ch2,
    input  wire signed [15:0] eeg_ch3,

    // ----- Primary outputs -----
    output wire        classification_valid,
    output wire        seizure_alert
);

    // =====================================================================
    // FSM States
    // =====================================================================
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_ENCODE = 3'd1;
    localparam [2:0] ST_LAYER1 = 3'd2;
    localparam [2:0] ST_LAYER2 = 3'd3;
    localparam [2:0] ST_DECODE = 3'd4;

    reg [2:0] state;
    reg layer1_enable;
    reg layer2_enable;
    reg decoder_enable;

    // =====================================================================
    // Internal wires
    // =====================================================================
    wire [63:0]  encoder_spike_bus;
    wire         encoder_valid;

    wire [7:0]   layer1_spikes;
    wire [127:0] layer1_membranes;
    wire [127:0] layer1_wsums;

    wire [0:0]   layer2_spikes;
    wire [15:0]  layer2_membranes;
    wire [15:0]  layer2_wsums;

    wire [15:0]  decoder_rate;
    wire         decoder_valid;
    wire         decoder_alert;

    // =====================================================================
    // FSM — Pipeline controller
    // =====================================================================
    always @(posedge clk_50m or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            layer1_enable  <= 1'b0;
            layer2_enable  <= 1'b0;
            decoder_enable <= 1'b0;
        end else begin
            // Default: all enables deasserted each cycle
            layer1_enable  <= 1'b0;
            layer2_enable  <= 1'b0;
            decoder_enable <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (sample_valid)
                        state <= ST_ENCODE;
                end

                ST_ENCODE: begin
                    if (encoder_valid) begin
                        layer1_enable <= 1'b1;   // pre-assert for next cycle
                        state <= ST_LAYER1;
                    end
                end

                ST_LAYER1: begin
                    // Layer1 neurons processed (enable was high from prev cycle)
                    layer2_enable <= 1'b1;   // pre-assert for next cycle
                    state <= ST_LAYER2;
                end

                ST_LAYER2: begin
                    // Layer2 neuron processed
                    decoder_enable <= 1'b1;  // pre-assert for next cycle
                    state <= ST_DECODE;
                end

                ST_DECODE: begin
                    // Decoder processed
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // =====================================================================
    // Encoder: 4-ch EEG → 64-bit one-hot spike bus
    // =====================================================================
    eeg_amplitude_encoder #(
        .SAMPLE_BITS(16),
        .CH0_MIN(ENC_CH0_MIN), .CH0_MAX(ENC_CH0_MAX),
        .CH1_MIN(ENC_CH1_MIN), .CH1_MAX(ENC_CH1_MAX),
        .CH2_MIN(ENC_CH2_MIN), .CH2_MAX(ENC_CH2_MAX),
        .CH3_MIN(ENC_CH3_MIN), .CH3_MAX(ENC_CH3_MAX)
    ) encoder (
        .clk         (clk_50m),
        .rst_n       (rst_n),
        .sample_valid(sample_valid),
        .eeg_ch0     (eeg_ch0),
        .eeg_ch1     (eeg_ch1),
        .eeg_ch2     (eeg_ch2),
        .eeg_ch3     (eeg_ch3),
        .spike_bus   (encoder_spike_bus),
        .spike_valid (encoder_valid)
    );

    // =====================================================================
    // Layer 1: 64 → 8 LIF neurons
    // =====================================================================
    dense_layer #(
        .INPUTS       (64),
        .NEURONS      (8),
        .WEIGHT_BITS  (8),
        .MEMBRANE_BITS(16),
        .THRESHOLD    (L1_THRESHOLD),
        .LEAK         (L1_LEAK),
        .DECAY_SHIFT  (L1_DECAY_SHIFT),
        .WEIGHT_FILE  (L1_WEIGHT_FILE)
    ) layer1 (
        .clk              (clk_50m),
        .rst_n            (rst_n),
        .enable           (layer1_enable),
        .spike_in         (encoder_spike_bus),
        .spike_out        (layer1_spikes),
        .membrane_flat    (layer1_membranes),
        .weighted_sum_flat(layer1_wsums)
    );

    // =====================================================================
    // Layer 2: 8 → 1 LIF neuron
    // =====================================================================
    dense_layer #(
        .INPUTS       (8),
        .NEURONS      (1),
        .WEIGHT_BITS  (8),
        .MEMBRANE_BITS(16),
        .THRESHOLD    (L2_THRESHOLD),
        .LEAK         (L2_LEAK),
        .DECAY_SHIFT  (L2_DECAY_SHIFT),
        .WEIGHT_FILE  (L2_WEIGHT_FILE)
    ) layer2 (
        .clk              (clk_50m),
        .rst_n            (rst_n),
        .enable           (layer2_enable),
        .spike_in         (layer1_spikes),
        .spike_out        (layer2_spikes),
        .membrane_flat    (layer2_membranes),
        .weighted_sum_flat(layer2_wsums)
    );

    // =====================================================================
    // Decoder: spike rate → seizure alert
    // =====================================================================
    firing_rate_decoder #(
        .WINDOW_SIZE     (DECODER_WINDOW),
        .POST_ICTAL_BLANK(DECODER_BLANKING),
        .FIRE_THRESHOLD  (DECODER_THRESHOLD),
        .RATE_BITS       (16)
    ) decoder (
        .clk                (clk_50m),
        .rst_n              (rst_n),
        .enable             (decoder_enable),
        .spike_in           (layer2_spikes[0]),
        .firing_rate        (decoder_rate),
        .seizure_detected   (decoder_alert),
        .classification_valid(decoder_valid)
    );

    // =====================================================================
    // Output assignments
    // =====================================================================
    assign classification_valid = decoder_valid;
    assign seizure_alert        = decoder_alert;

endmodule
