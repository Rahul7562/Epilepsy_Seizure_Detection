# ============================================================================
# Vivado XSIM simulation commands for SNN Seizure Detection
# ============================================================================
# Usage from Vivado Tcl console or command line:
#   cd <project_verilog_dir>
#   xvlog -sv lif_neuron.v dense_layer.v eeg_amplitude_encoder.v firing_rate_decoder.v snn_seizure_top.v snn_seizure_tb.v
#   xelab snn_seizure_tb -s snn_sim -debug typical
#   xsim snn_sim -runall -log xsim_run.log
# ============================================================================

# Step 1: Compile all Verilog sources
puts "=== Compiling Verilog sources ==="
exec xvlog lif_neuron.v dense_layer.v eeg_amplitude_encoder.v firing_rate_decoder.v snn_seizure_top.v snn_seizure_tb.v 2>@1

# Step 2: Elaborate
puts "=== Elaborating design ==="
exec xelab snn_seizure_tb -s snn_sim -debug typical 2>@1

# Step 3: Simulate
puts "=== Running simulation ==="
exec xsim snn_sim -runall -log xsim_run.log 2>@1

puts "=== Simulation complete. Check xsim_run.log and seizure_sim.vcd ==="
