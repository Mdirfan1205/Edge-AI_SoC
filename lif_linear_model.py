import numpy as np

class LinearLIFNeuron:
    """
    Digital LIF Neuron with Subtractive Leak, Zero-Floor Clamping, and Hard Reset.
    Rule per step:
      1. voltage = max(0, voltage + input - leak)
      2. if voltage >= threshold: spike = 1, voltage = 0
         else: spike = 0
    """
    def __init__(self, threshold: int, leak: int):
        self.threshold = np.int32(threshold)
        self.leak = np.int32(leak)
        self.voltage = np.int32(0)

    def reset(self):
        self.voltage = np.int32(0)

    def step(self, current_in: int):
        # 1. Integrate input and subtract constant leak
        v_next = self.voltage + np.int32(current_in) - self.leak

        # 2. Clamp at 0 (never below 0)
        if v_next < 0:
            v_next = np.int32(0)

        # 3. Threshold check & hard reset to 0
        if v_next >= self.threshold:
            spike = 1
            self.voltage = np.int32(0)
        else:
            spike = 0
            self.voltage = v_next

        return int(self.voltage), spike


def run_experiment():
    # Single deterministic input stream (15 time steps)
    # Includes silent gaps (0), weak inputs (2), and strong bursts (5, 6, 7)
    input_stream = [4, 5, 0, 0, 3, 6, 4, 0, 2, 2, 7, 5, 0, 6, 6]

    # Three requested configurations: (Label, Threshold, Leak)
    settings = [
        ("Setting 1 (Threshold=10, Leak=1)", 10, 1),
        ("Setting 2 (Threshold=15, Leak=1)", 15, 1),
        ("Setting 3 (Threshold=10, Leak=2)", 10, 2),
    ]

    output_filename = "lif_simulation_results.txt"

    with open(output_filename, "w") as f:
        header = "LIF NEURON GOLDEN REFERENCE SIMULATION\n" + "=" * 62 + "\n"
        header += f"Input Stream ({len(input_stream)} steps): {input_stream}\n\n"
        print(header, end="")
        f.write(header)

        for title, thresh, leak in settings:
            neuron = LinearLIFNeuron(threshold=thresh, leak=leak)
            total_spikes = 0

            section = f"{title}\n"
            section += "-" * 62 + "\n"
            section += f"{'Step':<6} | {'Input':<7} | {'Expected Voltage (Post-Step)':<30} | {'Spike':<5}\n"
            section += "-" * 62 + "\n"

            for t, inp in enumerate(input_stream):
                v_out, spk = neuron.step(inp)
                total_spikes += spk
                section += f"{t:<6} | {inp:<7} | {v_out:<30} | {spk:<5}\n"

            section += "-" * 62 + "\n"
            section += f"Total Spikes Fired: {total_spikes}\n\n"

            print(section, end="")
            f.write(section)

    print(f"Successfully saved all inputs, expected voltages, and spikes to '{output_filename}'.")


if __name__ == "__main__":
    run_experiment()
