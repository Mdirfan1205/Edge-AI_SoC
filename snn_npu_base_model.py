import numpy as np

class NPU_LIF_Layer:
    """
    Reusable Fixed-Point Edge AI Layer (INT8 MAC -> INT32 Acc -> Fixed LIF).
    Designed as a clean base template for future multi-layer expansion.
    """
    def __init__(self, weights_fp32, frac_bits=6, v_th_float=1.0, leak_shift=3):
        self.scale = 1 << frac_bits
        self.leak_shift = leak_shift
        self.v_th = np.int32(round(v_th_float * self.scale))

        # Quantize weights to INT8 [-128, 127]
        self.weights_int8 = np.clip(np.round(weights_fp32 * self.scale), -128, 127).astype(np.int8)
        self.num_neurons, self.num_inputs = self.weights_int8.shape
        self.v_mem = np.zeros(self.num_neurons, dtype=np.int32)

    def forward(self, input_spikes):
        # 1. NPU MAC (INT8 -> INT32 accumulator)
        acc_int32 = np.dot(self.weights_int8.astype(np.int32), input_spikes.astype(np.int32))

        # 2. Fixed-Point LIF Update (Shift Leak + Integrate)
        self.v_mem = (self.v_mem - (self.v_mem >> self.leak_shift)) + acc_int32

        # 3. Spike Generation & Reset-by-Subtraction
        spikes = (self.v_mem >= self.v_th).astype(np.uint8)
        self.v_mem = np.where(spikes == 1, self.v_mem - self.v_th, self.v_mem)

        return spikes, self.v_mem.copy()

# --- Minimal Usage Example ---
if __name__ == "__main__":
    np.random.seed(42)
    weights = np.random.uniform(-0.3, 0.8, size=(3, 4)).astype(np.float32) # 3 neurons, 4 inputs
    layer = NPU_LIF_Layer(weights_fp32=weights, frac_bits=6, v_th_float=1.0, leak_shift=3)

    # Single step execution
    sample_spikes = np.array([1, 0, 1, 0], dtype=np.uint8)
    out_spikes, v_mem = layer.forward(sample_spikes)

    print("INT8 Weights:\n", layer.weights_int8)
    print("Output Spikes:", out_spikes, "| INT32 V_mem:", v_mem)
