# ==============================================================================
# SNN Seizure Detection — Timing Constraints for Zynq-7020 (xc7z020clg484-1)
# ==============================================================================
# Target: 50 MHz system clock (20 ns period)
# ==============================================================================

# --- Primary Clock Definition -------------------------------------------------
# Defines a 50 MHz clock on the clk_50m input port
create_clock -period 20.000 -name clk_50m -waveform {0.000 10.000} [get_ports clk_50m]

# --- Input Delay Constraints --------------------------------------------------
# All input signals are synchronous to clk_50m.
# Assume worst-case 8 ns input delay (board trace + setup margin).
set_input_delay -clock clk_50m -max 8.000 [get_ports rst_n]
set_input_delay -clock clk_50m -min 1.000 [get_ports rst_n]

set_input_delay -clock clk_50m -max 8.000 [get_ports sample_valid]
set_input_delay -clock clk_50m -min 1.000 [get_ports sample_valid]

set_input_delay -clock clk_50m -max 8.000 [get_ports {eeg_ch0[*]}]
set_input_delay -clock clk_50m -min 1.000 [get_ports {eeg_ch0[*]}]

set_input_delay -clock clk_50m -max 8.000 [get_ports {eeg_ch1[*]}]
set_input_delay -clock clk_50m -min 1.000 [get_ports {eeg_ch1[*]}]

set_input_delay -clock clk_50m -max 8.000 [get_ports {eeg_ch2[*]}]
set_input_delay -clock clk_50m -min 1.000 [get_ports {eeg_ch2[*]}]

set_input_delay -clock clk_50m -max 8.000 [get_ports {eeg_ch3[*]}]
set_input_delay -clock clk_50m -min 1.000 [get_ports {eeg_ch3[*]}]

# --- Output Delay Constraints ------------------------------------------------
# Assume worst-case 8 ns output delay for downstream logic/board.
set_output_delay -clock clk_50m -max 8.000 [get_ports classification_valid]
set_output_delay -clock clk_50m -min 0.000 [get_ports classification_valid]

set_output_delay -clock clk_50m -max 8.000 [get_ports seizure_alert]
set_output_delay -clock clk_50m -min 0.000 [get_ports seizure_alert]

# --- False Path on Asynchronous Reset -----------------------------------------
# rst_n is an asynchronous active-low reset; exclude from timing analysis
set_false_path -from [get_ports rst_n]
