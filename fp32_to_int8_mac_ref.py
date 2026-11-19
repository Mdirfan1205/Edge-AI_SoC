import numpy as np

class FP32ToINT8MACReference:
    """
    Bit-Exact Reference Model for FP32 -> INT8 Quantized MAC with Hardware Requantization.
    Pipeline:
      1. Quantize FP32 inputs & weights to INT8 (Q1.6 format, scale = 2^6 = 64)
      2. Accumulate products (INT8 * INT8) into an INT32 accumulator (Q2.12 format)
      3. Requantize INT32 -> INT8 via Round-Half-Up, Arithmetic Right-Shift, and Saturation Clamping
    """
    def __init__(self, frac_bits: int = 6):
        self.frac_bits = frac_bits
        self.scale = 1 << frac_bits                  # 2^6 = 64
        self.half_lsb = np.int32(1 << (frac_bits - 1)) if frac_bits > 0 else np.int32(0)

    def quantize_fp32_to_int8(self, fp32_array: np.ndarray) -> np.ndarray:
        """Converts FP32 tensor into signed INT8 [-128, 127] using round-to-nearest."""
        scaled = np.round(fp32_array * self.scale)
        return np.clip(scaled, -128, 127).astype(np.int8)

    def mac_and_requantize(self, x_int8: np.ndarray, w_int8: np.ndarray):
        """
        Executes hardware-accurate MAC and INT32 -> INT8 requantization:
          out_int8 = clamp((acc_int32 + half_lsb) >> shift, -128, 127)
        """
        # 1. Widen operands to INT32 and accumulate (prevents 8-bit/16-bit overflow)
        acc_int32 = np.sum(x_int8.astype(np.int32) * w_int8.astype(np.int32), dtype=np.int32)

        # 2. Requantize: Add rounding bias (0.5 LSB) & arithmetic right-shift by frac_bits
        shifted = (acc_int32 + self.half_lsb) >> self.frac_bits

        # 3. Saturate to signed 8-bit output range [-128, 127]
        out_int8 = np.clip(shifted, -128, 127).astype(np.int8)

        return acc_int32, out_int8

    def fp32_golden_dot(self, x_fp32: np.ndarray, w_fp32: np.ndarray) -> float:
        """Ideal 32-bit floating-point dot product for accuracy comparison."""
        return float(np.dot(x_fp32, w_fp32))


if __name__ == "__main__":
    ref = FP32ToINT8MACReference(frac_bits=6)  # Q1.6 format (Scale = 64)

    # Test cases: (Description, FP32 Inputs, FP32 Weights)
    test_vectors = [
        ("Normal positive dot product",
         np.array([0.50,  0.25, -0.75,  0.50], dtype=np.float32),
         np.array([0.75,  0.50, -0.50,  0.25], dtype=np.float32)),

        ("Negative dominant dot product",
         np.array([-0.80, 0.40, -0.50,  0.20], dtype=np.float32),
         np.array([ 0.90, -0.75, 0.60, -0.50], dtype=np.float32)),

        ("Positive saturation (overflow clamp)",
         np.array([1.50,  1.50, -1.20,  1.00], dtype=np.float32),
         np.array([1.20,  1.00, -1.00,  0.80], dtype=np.float32)),
    ]

    print(f"{'Case':<34} | {'FP32 Ref':<9} | {'INT32 Acc':<10} | {'INT8 Out':<8} | {'Dequant FP32':<12} | {'Abs Err'}")
    print("-" * 98)

    for desc, x_f, w_f in test_vectors:
        # Ideal FP32 result
        fp32_out = ref.fp32_golden_dot(x_f, w_f)

        # Quantize FP32 -> INT8
        x_q = ref.quantize_fp32_to_int8(x_f)
        w_q = ref.quantize_fp32_to_int8(w_f)

        # Hardware MAC + Shift/Round Requantization
        acc_32, out_8 = ref.mac_and_requantize(x_q, w_q)

        # Convert INT8 output back to float to measure quantization error
        dequant_out = float(out_8) / ref.scale
        err = abs(fp32_out - dequant_out)

        print(f"{desc:<34} | {fp32_out:9.4f} | {acc_32:10d} | {out_8:8d} | {dequant_out:12.4f} | {err:.4f}")
