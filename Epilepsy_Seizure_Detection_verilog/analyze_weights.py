"""Analyze expected weighted sums with ONE-HOT encoding for threshold tuning."""
import os, random

def parse_hex_signed8(hex_str):
    val = int(hex_str.strip(), 16)
    if val >= 128:
        val -= 256
    return val

base = os.path.dirname(__file__)

with open(os.path.join(base, "layer1_weights.mem")) as f:
    l1_flat = [parse_hex_signed8(line) for line in f if line.strip()]

with open(os.path.join(base, "layer2_weights.mem")) as f:
    l2_flat = [parse_hex_signed8(line) for line in f if line.strip()]

print("=== ONE-HOT ENCODING ANALYSIS ===")
print("With one-hot: exactly 1 spike per channel, 4 total per timestep")
print()

# For each L1 neuron, calculate the weighted sum for every possible 
# combination of one-hot inputs (1 from each channel group of 16)
for n in range(8):
    weights = l1_flat[n*64 : (n+1)*64]
    ch_weights = [weights[ch*16:(ch+1)*16] for ch in range(4)]
    
    # All 16^4 = 65536 possible combinations
    all_sums = []
    for l0 in range(16):
        for l1 in range(16):
            for l2 in range(16):
                for l3 in range(16):
                    ws = ch_weights[0][l0] + ch_weights[1][l1] + ch_weights[2][l2] + ch_weights[3][l3]
                    all_sums.append(ws)
    
    pos_sums = [s for s in all_sums if s > 0]
    neg_sums = [s for s in all_sums if s < 0]
    
    print(f"L1 Neuron {n}:")
    print(f"  Weight sum range: [{min(all_sums):+d}, {max(all_sums):+d}]")
    print(f"  Mean: {sum(all_sums)/len(all_sums):+.1f}")
    print(f"  Positive sums: {len(pos_sums)}/{len(all_sums)} ({100*len(pos_sums)/len(all_sums):.1f}%)")
    
    # What thresholds would give reasonable firing rates?
    for thresh in [50, 100, 150, 200, 300, 400, 500]:
        exceed = sum(1 for s in all_sums if s >= thresh)
        pct = 100 * exceed / len(all_sums)
        print(f"    >{thresh:3d}: {pct:5.1f}% of inputs exceed")
    
    # With α=0.9375 and steady state gain = 16:
    # steady_state = weighted_sum / (1-0.9375) = weighted_sum * 16
    # But with soft reset, the effective gain is lower
    # The membrane will fire when accumulated sum >= threshold
    # Time to fire = threshold / avg_positive_weighted_sum (roughly)
    avg_pos = sum(pos_sums) / max(len(pos_sums), 1) if pos_sums else 0
    print(f"  Avg positive w_sum: {avg_pos:.1f}")
    print()

print("\n=== L2 ANALYSIS ===")
print(f"L2 weights: {l2_flat}")
print()

# L2 sees 8-bit spike_in from L1 (one bit per neuron)
# Which neurons fire and how often determines L2's input
for pattern_name, pattern in [
    ("N0,N1 only", 0b00000011),
    ("N0,N1,N5", 0b00100011),
    ("N0,N1,N5,N6", 0b01100011),
    ("N0,N1,N5,N6,N7", 0b11100011),
    ("All except N2", 0b11111011),
    ("N2 only", 0b00000100),
    ("None", 0b00000000),
]:
    ws = sum(l2_flat[i] for i in range(8) if pattern & (1 << i))
    print(f"  {pattern_name:25s} ({pattern:08b}): L2 w_sum = {ws:+d}")

# With steady state gain of ~16:
print()
print("Steady-state membrane (α=0.9375, gain≈16) for each L1→L2 pattern:")
for pattern_name, pattern in [
    ("N0,N1 only", 0b00000011),
    ("N0,N1,N5,N6", 0b01100011),
    ("None", 0b00000000),
]:
    ws = sum(l2_flat[i] for i in range(8) if pattern & (1 << i))
    ss = ws * 16  # steady state with α=0.9375
    print(f"  {pattern_name:25s}: w_sum={ws:+d}, steady_state≈{ss:+d}")
