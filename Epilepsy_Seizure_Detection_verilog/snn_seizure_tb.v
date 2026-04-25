`timescale 1ns/1ps

// =============================================================================
// SNN Seizure Detection Testbench — Full debug + VCD
// =============================================================================

module snn_seizure_tb;

    localparam integer CLK_HALF_NS      = 10;
    localparam integer MAX_SAMPLES      = 20000;
    localparam integer RUN_SAMPLES      = 20000;    // fewer for faster debug sim
    localparam integer SAMPLE_GAP_CYCLES = 8;      // gap between samples

    // ----- DUT signals -----
    reg clk;
    reg rst_n;
    reg sample_valid;
    reg signed [15:0] eeg_ch0, eeg_ch1, eeg_ch2, eeg_ch3;

    wire        classification_valid;
    wire        seizure_alert;
    wire [15:0] firing_rate_dbg     = dut.decoder_rate;
    wire [63:0] spike_bus_dbg       = dut.encoder_spike_bus;
    wire [7:0]  layer1_spikes_dbg   = dut.layer1_spikes;
    wire        layer2_spike_dbg    = dut.layer2_spikes[0];
    wire [127:0] layer1_membrane_dbg= dut.layer1_membranes;
    wire [127:0] layer1_wsum_dbg    = dut.layer1_wsums;
    wire [15:0]  layer2_membrane_dbg= dut.layer2_membranes;
    wire [15:0]  layer2_wsum_dbg    = dut.layer2_wsums;
    wire [2:0]   fsm_state_dbg      = dut.state;

    // ----- Per-neuron membrane extraction (for display) -----
    wire signed [15:0] l1_mem0 = layer1_membrane_dbg[15:0];
    wire signed [15:0] l1_mem1 = layer1_membrane_dbg[31:16];
    wire signed [15:0] l1_mem2 = layer1_membrane_dbg[47:32];
    wire signed [15:0] l1_mem3 = layer1_membrane_dbg[63:48];
    wire signed [15:0] l1_mem4 = layer1_membrane_dbg[79:64];
    wire signed [15:0] l1_mem5 = layer1_membrane_dbg[95:80];
    wire signed [15:0] l1_mem6 = layer1_membrane_dbg[111:96];
    wire signed [15:0] l1_mem7 = layer1_membrane_dbg[127:112];
    wire signed [15:0] l2_mem0 = layer2_membrane_dbg[15:0];

    wire signed [15:0] l1_ws0 = layer1_wsum_dbg[15:0];
    wire signed [15:0] l1_ws1 = layer1_wsum_dbg[31:16];

    // Count active input spikes for debug
    integer active_spike_count;
    integer k;
    always @* begin
        active_spike_count = 0;
        for (k = 0; k < 64; k = k + 1)
            active_spike_count = active_spike_count + spike_bus_dbg[k];
    end

    // =====================================================================
    // DUT instantiation
    // =====================================================================
    snn_seizure_top #(
        .L1_WEIGHT_FILE    ("layer1_weights.mem"),
        .L2_WEIGHT_FILE    ("layer2_weights.mem"),
        .L1_THRESHOLD      (200),
        .L1_LEAK           (1),
        .L1_DECAY_SHIFT    (4),       // α = 0.9375
        .L2_THRESHOLD      (100),
        .L2_LEAK           (0),
        .L2_DECAY_SHIFT    (4),       // α = 0.9375
        .DECODER_WINDOW    (4096),
        .DECODER_BLANKING  (256),
        .DECODER_THRESHOLD (1500)
    ) dut (
        .clk_50m           (clk),
        .rst_n             (rst_n),
        .sample_valid      (sample_valid),
        .eeg_ch0           (eeg_ch0),
        .eeg_ch1           (eeg_ch1),
        .eeg_ch2           (eeg_ch2),
        .eeg_ch3           (eeg_ch3),
        .classification_valid(classification_valid),
        .seizure_alert     (seizure_alert)
    );

    // Clock
    always #CLK_HALF_NS clk = ~clk;

    // =====================================================================
    // Dataset memory
    // =====================================================================
    reg signed [15:0] eeg_mem_flat [0:(MAX_SAMPLES*4)-1];
    reg label_mem [0:MAX_SAMPLES-1];
    reg label_fifo [0:MAX_SAMPLES+1023];

    integer i;
    integer fifo_wr, fifo_rd;
    reg eval_label;

    integer sent_samples;
    integer total_samples;
    integer correct_predictions, wrong_predictions;
    integer false_positives, seizure_detections;
    integer true_negatives, false_negatives;

    integer layer1_nonzero_events, layer2_nonzero_events;
    integer layer1_zero_cycles, layer2_zero_cycles;
    integer encoder_change_events;
    reg [63:0] prev_spike_bus;

    // =====================================================================
    // Push one EEG sample — sample_valid HIGH for exactly 1 clock
    // =====================================================================
    task automatic push_sample;
        input signed [15:0] ch0, ch1, ch2, ch3;
        input lbl;
        begin
            @(posedge clk);
            eeg_ch0 <= ch0;
            eeg_ch1 <= ch1;
            eeg_ch2 <= ch2;
            eeg_ch3 <= ch3;
            sample_valid <= 1'b1;

            label_fifo[fifo_wr] = lbl;
            fifo_wr = fifo_wr + 1;
            sent_samples = sent_samples + 1;

            @(posedge clk);
            sample_valid <= 1'b0;
            repeat (SAMPLE_GAP_CYCLES) @(posedge clk);
        end
    endtask

    // =====================================================================
    // Feed dataset from memory
    // =====================================================================
    task automatic feed_dataset;
        input integer sample_count;
        integer idx;
        begin
            for (idx = 0; idx < sample_count; idx = idx + 1) begin
                push_sample(
                    eeg_mem_flat[(idx*4)+0],
                    eeg_mem_flat[(idx*4)+1],
                    eeg_mem_flat[(idx*4)+2],
                    eeg_mem_flat[(idx*4)+3],
                    label_mem[idx]
                );
            end
        end
    endtask

    // =====================================================================
    // Feed alternating high/low pattern (guaranteed diverse spikes)
    // =====================================================================
    task automatic feed_alternating;
        input integer sample_count;
        integer idx;
        reg signed [15:0] hi, lo;
        begin
            hi = 16'sd20000;
            lo = -16'sd20000;
            for (idx = 0; idx < sample_count; idx = idx + 1) begin
                if (idx[0])
                    push_sample(lo, hi, lo, hi, 1'b1);
                else
                    push_sample(hi, lo, hi, lo, 1'b0);
            end
        end
    endtask

    // =====================================================================
    // Monitoring — track neuron activity
    // =====================================================================
    always @(posedge clk) begin
        if (!rst_n) begin
            prev_spike_bus        <= 64'd0;
            encoder_change_events <= 0;
            layer1_zero_cycles    <= 0;
            layer2_zero_cycles    <= 0;
            layer1_nonzero_events <= 0;
            layer2_nonzero_events <= 0;
        end else begin
            if (spike_bus_dbg != prev_spike_bus)
                encoder_change_events <= encoder_change_events + 1;
            prev_spike_bus <= spike_bus_dbg;

            if (layer1_spikes_dbg == 8'd0)
                layer1_zero_cycles <= layer1_zero_cycles + 1;
            else
                layer1_nonzero_events <= layer1_nonzero_events + 1;

            if (!layer2_spike_dbg)
                layer2_zero_cycles <= layer2_zero_cycles + 1;
            else
                layer2_nonzero_events <= layer2_nonzero_events + 1;

            // ---- Periodic detailed debug print ----
            if (layer1_spikes_dbg != 8'd0) begin
                $display("T=%0t [L1 FIRE] spikes=%08b mem0=%0d mem1=%0d ws0=%0d active_in=%0d",
                    $time, layer1_spikes_dbg, l1_mem0, l1_mem1, l1_ws0, active_spike_count);
            end

            if (layer2_spike_dbg) begin
                $display("T=%0t [L2 FIRE] l2_mem=%0d l1_spikes=%08b", $time, l2_mem0, layer1_spikes_dbg);
            end

            if (classification_valid) begin
                if (fifo_rd < fifo_wr)
                    eval_label = label_fifo[fifo_rd];
                else
                    eval_label = 1'b0;
                fifo_rd = fifo_rd + 1;

                total_samples = total_samples + 1;
                if (seizure_alert == eval_label)
                    correct_predictions = correct_predictions + 1;
                else
                    wrong_predictions = wrong_predictions + 1;

                if (seizure_alert && !eval_label)
                    false_positives = false_positives + 1;
                if (seizure_alert && eval_label)
                    seizure_detections = seizure_detections + 1;
                if (!seizure_alert && !eval_label)
                    true_negatives = true_negatives + 1;
                if (!seizure_alert && eval_label)
                    false_negatives = false_negatives + 1;

                if ((total_samples <= 20) || ((total_samples % 500) == 0) || seizure_alert) begin
                    $display("T=%0t | cls=%0b alert=%0b label=%0b rate=%0d l2=%0b l1=%08b mem0=%0d",
                        $time, classification_valid, seizure_alert, eval_label,
                        firing_rate_dbg, layer2_spike_dbg, layer1_spikes_dbg, l1_mem0);
                end
            end
        end
    end

    // =====================================================================
    // Main stimulus
    // =====================================================================
    initial begin
        $dumpfile("seizure_sim.vcd");
        $dumpvars(0, snn_seizure_tb);

        $readmemh("eeg_dataset.mem", eeg_mem_flat, 0, (MAX_SAMPLES*4)-1);
        $readmemb("labels.mem", label_mem, 0, MAX_SAMPLES-1);

        clk          = 0;
        rst_n        = 0;
        sample_valid = 0;
        eeg_ch0 = 0; eeg_ch1 = 0; eeg_ch2 = 0; eeg_ch3 = 0;

        fifo_wr = 0; fifo_rd = 0; eval_label = 0; sent_samples = 0;
        total_samples = 0;
        correct_predictions = 0; wrong_predictions = 0;
        false_positives = 0; seizure_detections = 0;
        true_negatives = 0; false_negatives = 0;

        // Reset
        #100;
        rst_n = 1;
        #20;

        $display("===== SNN SEIZURE DETECTION SIMULATION START =====");
        $display("  Encoding: ONE-HOT (4 active spikes per sample)");
        $display("  L1: threshold=%0d, decay_shift=%0d (alpha=0.9375)", 200, 4);
        $display("  L2: threshold=%0d, decay_shift=%0d (alpha=0.9375)", 100, 4);
        $display("  Decoder: window=%0d, threshold=%0d, blanking=%0d", 4096, 1500, 256);

        // Feed real EEG dataset
        feed_dataset(RUN_SAMPLES);

        // Feed alternating pattern (stress test)
        feed_alternating(200);

        repeat (512) @(posedge clk);

        // =====================================================================
        // FINAL REPORT
        // =====================================================================
        $display("=================================================");
        $display("            FINAL SIMULATION REPORT              ");
        $display("=================================================");
        $display("Stimulus Samples Sent:     %0d", sent_samples);
        $display("Total Classified Samples:  %0d", total_samples);
        $display("Correct Predictions:       %0d", correct_predictions);
        $display("Wrong Predictions:         %0d", wrong_predictions);
        if (total_samples > 0)
            $display("Accuracy:                  %0f %%", (correct_predictions * 100.0) / total_samples);
        else
            $display("Accuracy:                  N/A");
        $display("True Positives (TP):       %0d", seizure_detections);
        $display("True Negatives (TN):       %0d", true_negatives);
        $display("False Positives (FP):      %0d", false_positives);
        $display("False Negatives (FN):      %0d", false_negatives);
        if (seizure_detections + false_negatives > 0)
            $display("Sensitivity (TPR):         %0f %%", (seizure_detections * 100.0) / (seizure_detections + false_negatives));
        if (true_negatives + false_positives > 0)
            $display("Specificity (TNR):         %0f %%", (true_negatives * 100.0) / (true_negatives + false_positives));
        $display("-------------------------------------------------");
        $display("Encoder change events:     %0d", encoder_change_events);
        $display("Layer1 nonzero events:     %0d", layer1_nonzero_events);
        $display("Layer1 zero cycles:        %0d", layer1_zero_cycles);
        $display("Layer2 nonzero events:     %0d", layer2_nonzero_events);
        $display("Layer2 zero cycles:        %0d", layer2_zero_cycles);
        if (sent_samples > 0) begin
            $display("L1 firing rate:            %0f %%", (layer1_nonzero_events * 100.0) / (layer1_nonzero_events + layer1_zero_cycles));
            $display("L2 firing rate:            %0f %%", (layer2_nonzero_events * 100.0) / (layer2_nonzero_events + layer2_zero_cycles));
        end
        $display("-------------------------------------------------");

        if (layer1_nonzero_events == 0)
            $display("TB_FAIL: Layer1 neurons NEVER fired!");
        else
            $display("TB_OK:   Layer1 fired %0d times", layer1_nonzero_events);

        if (layer2_nonzero_events == 0)
            $display("TB_FAIL: Layer2 neuron NEVER fired!");
        else
            $display("TB_OK:   Layer2 fired %0d times", layer2_nonzero_events);

        if (seizure_detections == 0 && total_samples > 100)
            $display("TB_WARN: No seizures detected (may need tuning)");

        if (total_samples == 0)
            $display("TB_FAIL: classification_valid never asserted!");
        else
            $display("TB_PASS: classification stream active.");

        $display("=================================================");

        #100;
        $finish;
    end

endmodule
