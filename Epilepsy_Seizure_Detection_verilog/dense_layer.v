`timescale 1ns / 1ps

// =============================================================================
// Dense Layer — N LIF neurons with flat weight memory + direct $readmemh
// =============================================================================
// Key fixes:
//   • $readmemh called DIRECTLY with parameter string (not via task variable)
//   • Weights are properly mapped: weight_flat_init[neuron*INPUTS + input]
//   • Debug: per-neuron membrane + weighted_sum exposed
//   • Neuron interface updated for LIF with multiplicative decay
// =============================================================================

module dense_layer #(
    parameter integer INPUTS          = 64,
    parameter integer NEURONS         = 8,
    parameter integer WEIGHT_BITS     = 8,
    parameter integer MEMBRANE_BITS   = 16,
    parameter integer THRESHOLD       = 150,
    parameter integer LEAK            = 1,
    parameter integer DECAY_SHIFT     = 4,    // α = 1 - 2^(-DECAY_SHIFT)
    parameter         WEIGHT_FILE     = "layer1_weights.mem"
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               enable,
    input  wire [INPUTS-1:0]                  spike_in,
    output wire [NEURONS-1:0]                 spike_out,
    output wire [NEURONS*MEMBRANE_BITS-1:0]   membrane_flat,
    output wire [NEURONS*MEMBRANE_BITS-1:0]   weighted_sum_flat
);

    // ---- Weight storage ----
    reg signed [WEIGHT_BITS-1:0] weight_mem [0:NEURONS-1][0:INPUTS-1];
    reg signed [WEIGHT_BITS-1:0] weight_flat_init [0:(NEURONS*INPUTS)-1];

    // ---- Bias (zero for now, can be loaded later) ----
    reg signed [WEIGHT_BITS-1:0] bias_mem [0:NEURONS-1];

    // ---- Per-neuron packed weight bus ----
    wire [INPUTS*WEIGHT_BITS-1:0] neuron_weight_flat [0:NEURONS-1];

    // ---- Per-neuron outputs ----
    wire signed [MEMBRANE_BITS-1:0] membrane_out     [0:NEURONS-1];
    wire signed [MEMBRANE_BITS-1:0] wsum_out         [0:NEURONS-1];
    wire [31:0]                     spike_count_out   [0:NEURONS-1];
    wire [NEURONS-1:0]              neuron_spikes;

    integer n, ii;

    // =====================================================================
    // Weight initialization — DIRECT $readmemh, no task-variable indirection
    // =====================================================================
    initial begin
        // Zero everything first
        for (n = 0; n < NEURONS; n = n + 1) begin
            bias_mem[n] = {WEIGHT_BITS{1'b0}};
            for (ii = 0; ii < INPUTS; ii = ii + 1) begin
                weight_mem[n][ii] = {WEIGHT_BITS{1'b0}};
                weight_flat_init[(n*INPUTS) + ii] = {WEIGHT_BITS{1'b0}};
            end
        end

        // Load weights from file (direct parameter — avoids task-variable issues)
`ifndef SYNTHESIS
        $readmemh(WEIGHT_FILE, weight_flat_init);
        // Map flat array → 2D weight memory
        for (n = 0; n < NEURONS; n = n + 1) begin
            for (ii = 0; ii < INPUTS; ii = ii + 1) begin
                weight_mem[n][ii] = weight_flat_init[(n * INPUTS) + ii];
            end
        end
        $display("[dense_layer] Loaded weights from %s (%0d entries)", WEIGHT_FILE, NEURONS*INPUTS);
`endif
    end

    // =====================================================================
    // Pack 2D weight_mem into flat bus per neuron for lif_neuron ports
    // =====================================================================
    genvar gw, gi;
    generate
        for (gw = 0; gw < NEURONS; gw = gw + 1) begin : PACK_WEIGHTS
            for (gi = 0; gi < INPUTS; gi = gi + 1) begin : PACK_ELEM
                assign neuron_weight_flat[gw][(gi*WEIGHT_BITS) +: WEIGHT_BITS] = weight_mem[gw][gi];
            end
        end
    endgenerate

    // =====================================================================
    // Instantiate N LIF neurons
    // =====================================================================
    genvar gn;
    generate
        for (gn = 0; gn < NEURONS; gn = gn + 1) begin : GEN_NEURON
            lif_neuron #(
                .INPUTS        (INPUTS),
                .WEIGHT_BITS   (WEIGHT_BITS),
                .MEMBRANE_BITS (MEMBRANE_BITS),
                .THRESHOLD     (THRESHOLD),
                .LEAK          (LEAK),
                .DECAY_SHIFT   (DECAY_SHIFT)
            ) u_neuron (
                .clk            (clk),
                .rst_n          (rst_n),
                .enable         (enable),
                .spike_in       (spike_in),
                .weight_flat    (neuron_weight_flat[gn]),
                .bias           (bias_mem[gn]),
                .spike_out      (neuron_spikes[gn]),
                .membrane       (membrane_out[gn]),
                .weighted_sum_dbg(wsum_out[gn]),
                .spike_count    (spike_count_out[gn])
            );

            assign membrane_flat[(gn*MEMBRANE_BITS) +: MEMBRANE_BITS]     = membrane_out[gn];
            assign weighted_sum_flat[(gn*MEMBRANE_BITS) +: MEMBRANE_BITS]  = wsum_out[gn];
        end
    endgenerate

    assign spike_out = neuron_spikes;

endmodule
