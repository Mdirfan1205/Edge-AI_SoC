import os
import numpy as np

class EightInputDotRequantModel:
    """
    Software Reference Model for an 8-Input Dot Product + Requantization Unit:
      Step 1: 8-element dot product in INT32 (prevents intermediate overflow)
      Step 2: Symmetric rounding + right-shift by 'shift' bits (scales back down)
      Step 3: Saturation clamping to [-128, 127] (fits into INT8 output)
    """
    def __init__(self, shift: int = 6):
        self.shift = int(shift)
        # Half of 2^shift used for rounding (e.g., for shift=6, 2^6=64 -> round_offset=32)
        self.round_offset = np.int32(1 << (self.shift - 1)) if self.shift > 0 else np.int32(0)

    def dot_product_int32(self, inputs_int8: np.ndarray, weights_int8: np.ndarray) -> np.int32:
        """Multiplies 8 INT8 inputs with 8 INT8 weights and sums them in INT32."""
        x32 = inputs_int8.astype(np.int32)
        w32 = weights_int8.astype(np.int32)
        return np.sum(x32 * w32, dtype=np.int32)

    def requantize_to_int8(self, acc_int32: np.int32):
        """Rounds, shifts, and clamps an INT32 accumulator value into signed INT8."""
        if self.shift > 0:
            if acc_int32 >= 0:
                shifted = (acc_int32 + self.round_offset) >> self.shift
            else:
                shifted = -((-acc_int32 + self.round_offset) >> self.shift)
        else:
            shifted = acc_int32

        # Clamp strictly between -128 and +127
        clamped_int8 = np.clip(shifted, -128, 127).astype(np.int8)
        return int(shifted), int(clamped_int8)

    def forward(self, inputs_int8: np.ndarray, weights_int8: np.ndarray):
        """Runs full 8-input dot product followed by requantization."""
        acc_32 = self.dot_product_int32(inputs_int8, weights_int8)
        unclipped, out_8 = self.requantize_to_int8(acc_32)
        return int(acc_32), unclipped, out_8


def run_verification():
    # Scale = 64 (shift = 6), round_offset = 32
    model = EightInputDotRequantModel(shift=6)

    # Fixed 8 weights (INT8)
    weights = np.array([32, -16, 48, 24, -40, 12, 64, -20], dtype=np.int8)

    # 6 test cases testing normal math, half-way rounding, and saturation
    test_cases = [
        ("Normal positive sum",         np.array([10, -5, 12, 8, -4, 15, 6, -10], dtype=np.int8)),
        ("Normal negative sum",         np.array([-15, 20, -10, -12, 18, -8, -14, 10], dtype=np.int8)),
        ("Positive half-LSB (+0.5->1)", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)),
        ("Negative half-LSB (-0.5->-1)",np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)),
        ("Positive saturation (+127)",  np.array([60, -50, 70, 50, -60, 40, 80, -50], dtype=np.int8)),
        ("Negative saturation (-128)",  np.array([-60, 50, -70, -50, 60, -40, -80, 50], dtype=np.int8)),
    ]

    output_path = os.path.expanduser("~/edge_ai_soc_lab/mac_8input_results.txt")

    with open(output_path, "w") as f:
        header = (
            "8-INPUT DOT-PRODUCT & REQUANTIZATION GOLDEN REFERENCE\n"
            + "=" * 98 + "\n"
            f"Weights (8x INT8): {weights.tolist()} | Shift: >> {model.shift} (Divide by 64)\n"
            + "-" * 98 + "\n"
            f"{'Step':<5} | {'Test Case':<27} | {'INT32 Sum':<11} | {'After Shift':<11} | {'INT8 Out':<8} | {'Status'}\n"
            + "-" * 98 + "\n"
        )
        print(header, end="")
        f.write(header)

        for step, (name, x_in) in enumerate(test_cases):
            acc_32, shifted, out_8 = model.forward(x_in, weights)
            expected = int(np.clip(shifted, -128, 127))
            status = "PASS" if out_8 == expected else "FAIL"

            row = f"{step:<5} | {name:<27} | {acc_32:<11} | {shifted:<11} | {out_8:<8} | {status}\n"
            print(row, end="")
            f.write(row)

        footer = "-" * 98 + "\n"
        print(footer, end="")
        f.write(footer)

    print(f"Saved results to '{output_path}'.")


if __name__ == "__main__":
    run_verification()
