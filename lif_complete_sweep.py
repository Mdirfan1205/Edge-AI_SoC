import numpy as np

class HardwareLinearLIF:
    """
    Bit-Exact Linear LIF Neuron Model with:
      1. Fixed-point round-half-up scaling (frac_bits)
      2. Linear integration & subtractive leak: V = V + I_rounded - leak
      3. Zero-floor underflow clamping (never < 0)
      4. Register bit-width overflow saturation (never > max_val)
      5. Threshold firing (>= threshold) & hard reset to 0
    """
    def __init__(self, threshold: int, leak: int, bit_width: int = 8, frac_bits: int = 2):
        self.threshold = np.int32(threshold)
        self.leak = np.int32(leak)
        self.frac_bits = frac_bits
        self.half_lsb = np.int32(1 << (frac_bits - 1)) if frac_bits > 0 else np.int32(0)
        self.max_val = np.int32((1 << bit_width) - 1)  # 8-bit unsigned max = 255
        self.voltage = np.int32(0)

    def round_input(self, raw_input: int) -> np.int32:
        """Scales higher-precision Q(m.n) input down to register resolution using round-half-up."""
        val = np.int32(raw_input)
        if self.frac_bits == 0:
            return val
        return (val + self.half_lsb) >> self.frac_bits

    def step(self, raw_input: int):
        # 1. Fixed-point rounding of incoming current
        rounded_in = self.round_input(raw_input)

        # 2. Linear integrate and subtract leak
        v_temp = self.voltage + rounded_in - self.leak

        # 3. Underflow (zero-floor) & Overflow (register saturation) clamping
        overflow = 0
        if v_temp < 0:
            v_clamped = np.int32(0)
        elif v_temp > self.max_val:
            v_clamped = self.max_val
            overflow = 1
        else:
            v_clamped = v_temp

        # 4. Threshold comparison & hard reset to 0
        if v_clamped >= self.threshold:
            spike = 1
            self.voltage = np.int32(0)
        else:
            spike = 0
            self.voltage = v_clamped

        return int(rounded_in), int(self.voltage), spike, overflow


def run_all_settings():
    # Single input stream in Q(m.2) format (scaled by 2^2 = 4)
    # Tests fractional rounding (14 -> 3.5 -> 4, 18 -> 4.5 -> 5), silent gaps (0),
    # weak inputs (6 -> 1.5 -> 2), and an extreme burst (1100 -> 275 -> saturates at 255)
    raw_input_stream = [14, 18, 0, 0, 10, 22, 16, 0, 6, 6, 26, 18, 0, 22, 1100]

    settings = [
        ("Setting 1 (Threshold=10, Leak=1)", 10, 1),
        ("Setting 2 (Threshold=15, Leak=1)", 15, 1),
        ("Setting 3 (Threshold=10, Leak=2)", 10, 2),
    ]

    output_filename = "lif_complete_results.txt"

    with open(output_filename, "w") as f:
        header = (
            "COMPLETE LINEAR LIF GOLDEN REFERENCE (WITH ROUNDING & 8-BIT OVERFLOW)\n"
            + "=" * 82 + "\n"
            f"Hardware Config: 8-bit Register (Max=255) | Frac_Bits=2 (Scale=4, Round-Half-Up)\n"
            f"Raw Input Stream ({len(raw_input_stream)} steps): {raw_input_stream}\n\n"
        )
        print(header, end="")
        f.write(header)

        for title, thresh, leak in settings:
            neuron = HardwareLinearLIF(threshold=thresh, leak=leak, bit_width=8, frac_bits=2)
            total_spikes = 0
            total_overflows = 0

            section = f"{title}\n"
            section += "-" * 82 + "\n"
            section += f"{'Step':<5} | {'Raw In':<7} | {'Rounded In':<10} | {'Expected Voltage':<17} | {'Spike':<6} | {'Overflow'}\n"
            section += "-" * 82 + "\n"

            for t, raw_in in enumerate(raw_input_stream):
                r_in, v_out, spk, ovf = neuron.step(raw_in)
                total_spikes += spk
                total_overflows += ovf
                section += f"{t:<5} | {raw_in:<7} | {r_in:<10} | {v_out:<17} | {spk:<6} | {ovf}\n"

            section += "-" * 82 + "\n"
            section += f"Summary -> Total Spikes: {total_spikes} | Overflow Events Saturated: {total_overflows}\n\n"

            print(section, end="")
            f.write(section)

    print(f"Saved complete verification table to '{output_filename}'.")


if __name__ == "__main__":
    run_all_settings()
