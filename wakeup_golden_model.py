import os
from collections import deque
import numpy as np

class WakeUpUnitGoldenModel:
    """
    Bit-Exact Hardware Wake-Up Logic Reference Model:
      - Maintains a W-bit shift register (sliding window) of binary spikes
      - Updates running spike count C[t] per clock cycle
      - Asserts wake = 1 when C[t] > limit ("passes a limit"), else 0
    """
    def __init__(self, window_size: int, spike_limit: int):
        self.window_size = int(window_size)
        self.spike_limit = int(spike_limit)
        # Shift register initialized to all 0s (hardware reset state)
        self.window = deque([0] * self.window_size, maxlen=self.window_size)
        self.current_count = 0

    def reset(self):
        self.window = deque([0] * self.window_size, maxlen=self.window_size)
        self.current_count = 0

    def step(self, spike_in: int):
        spike_bit = int(spike_in) & 1

        # Hardware incremental counter update:
        # Subtract oldest bit shifting out, add newest bit shifting in
        oldest_bit = self.window[0]
        self.window.append(spike_bit)
        self.current_count = self.current_count - oldest_bit + spike_bit

        # Wake condition: count passes the limit (> spike_limit)
        wake_out = 1 if self.current_count > self.spike_limit else 0

        return list(self.window), self.current_count, wake_out


def run_wakeup_suite(stream_name: str, spike_stream: list, configs: list, output_path: str):
    with open(output_path, "w") as f:
        header = (
            f"WAKE-UP LOGIC GOLDEN REFERENCE SIMULATION ({stream_name.upper()} INPUT)\n"
            + "=" * 92 + "\n"
            f"Input Spike Stream ({len(spike_stream)} cycles): {spike_stream}\n\n"
        )
        print(header, end="")
        f.write(header)

        for title, win_size, limit in configs:
            unit = WakeUpUnitGoldenModel(window_size=win_size, spike_limit=limit)
            total_wake_cycles = 0

            section = f"{title} [Window Size W={win_size} | Wake Condition: Count > {limit}]\n"
            section += "-" * 92 + "\n"
            section += f"{'Step':<5} | {'Spike In':<8} | {'Window Buffer (Old->New)':<26} | {'Window Count':<12} | {'Expected Wake'}\n"
            section += "-" * 92 + "\n"

            for t, spk in enumerate(spike_stream):
                win_state, count, wake = unit.step(spk)
                total_wake_cycles += wake
                win_str = str(win_state)
                section += f"{t:<5} | {spk:<8} | {win_str:<26} | {count:<12} | {wake}\n"

            section += "-" * 92 + "\n"
            section += f"Summary -> Total Wake Asserted Cycles: {total_wake_cycles}/{len(spike_stream)}\n\n"

            print(section, end="")
            f.write(section)

    print(f"Saved standardized test file: '{output_path}'\n")


if __name__ == "__main__":
    # 1. Test Input Streams (16 clock cycles each)
    # Quiet input: sparse isolated spikes (background noise)
    quiet_input_stream = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    # Busy input: dense spike bursts (active event)
    busy_input_stream  = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0]

    # 2. Two Window Size Configurations: (Label, Window_Size, Spike_Limit)
    window_configs = [
        ("Setting 1 (Short Window)", 4, 2),  # W=4 cycles, wakes when Count > 2 (i.e., 3 or 4 spikes)
        ("Setting 2 (Long Window) ", 8, 4),  # W=8 cycles, wakes when Count > 4 (i.e., 5+ spikes)
    ]

    base_dir = os.path.expanduser("~/edge_ai_soc_lab")
    quiet_file = os.path.join(base_dir, "wakeup_quiet_results.txt")
    busy_file  = os.path.join(base_dir, "wakeup_busy_results.txt")

    run_wakeup_suite("Quiet", quiet_input_stream, window_configs, quiet_file)
    run_wakeup_suite("Busy", busy_input_stream, window_configs, busy_file)
