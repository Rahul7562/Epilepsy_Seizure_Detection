`timescale 1ns / 1ps

// =============================================================================
// LIF Neuron — Leaky Integrate-and-Fire with configurable decay
// =============================================================================
// Two modes controlled by DECAY_SHIFT parameter:
// • DECAY_SHIFT > 0: SNN mode with multiplicative decay α = 1 - 2^(-DECAY_SHIFT)
//   membrane = α * membrane + weighted_sum
//   if membrane >= THRESHOLD → spike, soft reset (subtract THRESHOLD)
//
// • DECAY_SHIFT = 0: Stateless mode (mimics ANN ReLU behavior)
//   membrane = weighted_sum  (no accumulation — each step independent)
//   if membrane >= THRESHOLD → spike, reset to 0
//
// Saturation clamp to signed 16-bit range [-32768, +32767]
// Debug outputs: weighted_sum_dbg, spike_count
// =============================================================================

module lif_neuron #(
    parameter integer INPUTS         = 64,
    parameter integer WEIGHT_BITS    = 8,
    parameter integer MEMBRANE_BITS  = 16,
    parameter integer THRESHOLD      = 150,   // Calibrated for one-hot encoding
    parameter integer LEAK           = 1,     // Unused (kept for port compat)
    parameter integer DECAY_SHIFT    = 4      // 0 = stateless, >0 = LIF with α
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  enable,
    input  wire [INPUTS-1:0]                     spike_in,
    input  wire signed [INPUTS*WEIGHT_BITS-1:0]  weight_flat,
    input  wire signed [WEIGHT_BITS-1:0]         bias,
    output reg                                   spike_out,
    output reg  signed [MEMBRANE_BITS-1:0]       membrane,
    output reg  signed [MEMBRANE_BITS-1:0]       weighted_sum_dbg,
    output reg  [31:0]                           spike_count
);

    // Wide accumulator to prevent overflow during summation
    localparam integer ACC_BITS = MEMBRANE_BITS + WEIGHT_BITS + 8;

    integer i;
    reg signed [WEIGHT_BITS-1:0]    weight_i;
    reg signed [ACC_BITS-1:0]       weighted_sum;
    reg signed [ACC_BITS-1:0]       membrane_decayed;
    reg signed [ACC_BITS-1:0]       membrane_next;

    // -------------------------------------------------------------------------
    // Saturation function — clamp to signed MEMBRANE_BITS range
    // -------------------------------------------------------------------------
    function signed [MEMBRANE_BITS-1:0] saturate;
        input signed [ACC_BITS-1:0] val;
        reg signed [ACC_BITS-1:0] pos_max;
        reg signed [ACC_BITS-1:0] neg_min;
        begin
            pos_max = (1 <<< (MEMBRANE_BITS - 1)) - 1;   // +32767
            neg_min = -(1 <<< (MEMBRANE_BITS - 1));       // -32768
            if (val > pos_max)
                saturate = pos_max[MEMBRANE_BITS-1:0];
            else if (val < neg_min)
                saturate = neg_min[MEMBRANE_BITS-1:0];
            else
                saturate = val[MEMBRANE_BITS-1:0];
        end
    endfunction

    // -------------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_out       <= 1'b0;
            membrane        <= {MEMBRANE_BITS{1'b0}};
            weighted_sum_dbg <= {MEMBRANE_BITS{1'b0}};
            spike_count     <= 32'd0;
        end else if (enable) begin
            // --- Step 1: compute weighted sum of active input spikes ---------
            weighted_sum = $signed({{(ACC_BITS-WEIGHT_BITS){bias[WEIGHT_BITS-1]}}, bias});
            for (i = 0; i < INPUTS; i = i + 1) begin
                weight_i = weight_flat[(i*WEIGHT_BITS) +: WEIGHT_BITS];
                if (spike_in[i]) begin
                    weighted_sum = weighted_sum + $signed({{(ACC_BITS-WEIGHT_BITS){weight_i[WEIGHT_BITS-1]}}, weight_i});
                end
            end

            // Store debug value (saturated to 16-bit for display)
            weighted_sum_dbg <= saturate(weighted_sum);

            // --- Step 2: membrane update ---
            if (DECAY_SHIFT == 0) begin
                // Stateless mode: no accumulation, just use weighted_sum directly
                membrane_next = weighted_sum;
            end else begin
                // LIF mode: multiplicative decay then integrate
                // membrane_decayed = membrane * (1 - 2^-DECAY_SHIFT)
                membrane_decayed = $signed({{(ACC_BITS-MEMBRANE_BITS){membrane[MEMBRANE_BITS-1]}}, membrane})
                                 - ($signed({{(ACC_BITS-MEMBRANE_BITS){membrane[MEMBRANE_BITS-1]}}, membrane}) >>> DECAY_SHIFT);
                membrane_next = membrane_decayed + weighted_sum;
            end

            // Clamp negative membrane to zero (rectified LIF)
            if (membrane_next < 0)
                membrane_next = 0;

            // --- Step 3: threshold comparison and spike generation -----------
            if (membrane_next >= THRESHOLD) begin
                spike_out   <= 1'b1;
                if (DECAY_SHIFT == 0) begin
                    // Stateless: always reset to 0
                    membrane <= {MEMBRANE_BITS{1'b0}};
                end else begin
                    // Soft reset: subtract threshold, keep residual (per paper Eq. 3)
                    membrane <= saturate(membrane_next - THRESHOLD);
                end
                spike_count <= spike_count + 1'b1;
            end else begin
                spike_out   <= 1'b0;
                membrane    <= saturate(membrane_next);
            end
        end else begin
            spike_out <= 1'b0;
            // membrane RETAINS its value when not enabled (persistence)
        end
    end

endmodule
