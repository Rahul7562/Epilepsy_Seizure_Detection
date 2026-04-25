@echo off
REM Run SNN simulation using Vivado XSIM tools
cd /d "%~dp0"

echo === Step 1: Compile Verilog sources ===
call "C:\AMDDesignTools\2025.2\Vivado\bin\xvlog.bat" lif_neuron.v dense_layer.v eeg_amplitude_encoder.v firing_rate_decoder.v snn_seizure_top.v snn_seizure_tb.v -log xvlog_new.log
if errorlevel 1 (
    echo COMPILE FAILED - check xvlog_new.log
    type xvlog_new.log
    exit /b 1
)

echo === Step 2: Elaborate ===
call "C:\AMDDesignTools\2025.2\Vivado\bin\xelab.bat" snn_seizure_tb -s snn_sim_new -debug typical -relax -log xelab_new.log
if errorlevel 1 (
    echo ELABORATE FAILED - check xelab_new.log
    type xelab_new.log
    exit /b 1
)

echo === Step 3: Simulate ===
call "C:\AMDDesignTools\2025.2\Vivado\bin\xsim.bat" snn_sim_new -runall -log xsim_new.log
if errorlevel 1 (
    echo SIMULATE FAILED - check xsim_new.log
    type xsim_new.log
    exit /b 1
)

echo === DONE - check xsim_new.log for results ===
