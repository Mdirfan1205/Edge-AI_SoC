import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
import snntorch as snn
import tensorflow as tf

# ==============================================================================
# 1. HARDWARE FIXED-POINT PARAMETERS
# ==============================================================================
NUM_INPUTS = 4          # 4 input spike channels
NUM_HIDDEN = 3          # 3 LIF neurons in the hidden layer
TIME_STEPS = 10         # 10 simulation cycles

FRAC_BITS = 6           # Q1.6 fixed-point format
SCALE = 1 << FRAC_BITS  # Scale = 64
LEAK_SHIFT = 3          # beta = 1 - (1 / 2^3) = 0.875
V_TH_FLOAT = 1.0
V_TH_INT32 = 64         # 1.0 * 64

np.random.seed(42)

# Float weights & binary input spikes
weights_fp32 = np.random.uniform(-0.3, 0.8, size=(NUM_HIDDEN, NUM_INPUTS)).astype(np.float32)
input_spikes = (np.random.rand(TIME_STEPS, NUM_INPUTS) > 0.5).astype(np.uint8)

# Quantize FP32 weights -> INT8
weights_int8 = np.clip(np.round(weights_fp32 * SCALE), -128, 127).astype(np.int8)

# ==============================================================================
# 2. SOFTWARE GOLDEN MODEL: INT8 MAC + INT32 LIF HIDDEN LAYER
# ==============================================================================
class LIFHiddenLayerGolden:
    def __init__(self, weights_int8, v_th_int32, leak_shift):
        self.weights = weights_int8
        self.num_neurons, self.num_inputs = weights_int8.shape
        self.v_mem = np.zeros(self.num_neurons, dtype=np.int32)
        self.v_th = np.int32(v_th_int32)
        self.leak_shift = leak_shift

    def step(self, spikes_in):
        # Stage 1: NPU INT8 MAC Engine (Acc = Acc + A * B in INT32)
        acc_int32 = np.zeros(self.num_neurons, dtype=np.int32)
        for n in range(self.num_neurons):
            for i in range(self.num_inputs):
                acc_int32[n] += np.int32(spikes_in[i]) * np.int32(self.weights[n, i])

        # Stage 2: Fixed-Point LIF Update
        v_leaked = self.v_mem - (self.v_mem >> self.leak_shift) # V * 0.875
        self.v_mem = v_leaked + acc_int32                       # Integrate current

        # Stage 3: Spike Comparator & Reset-by-Subtraction
        spikes_out = (self.v_mem >= self.v_th).astype(np.uint8)
        self.v_mem = np.where(spikes_out == 1, self.v_mem - self.v_th, self.v_mem)

        return spikes_out, self.v_mem.copy()

# ==============================================================================
# 3. RUN & COMPARE AGAINST SNNTORCH & TENSORFLOW
# ==============================================================================
golden_layer = LIFHiddenLayerGolden(weights_int8, V_TH_INT32, LEAK_SHIFT)

# snnTorch reference
snn_lif = snn.Leaky(beta=0.875, threshold=V_TH_FLOAT, reset_mechanism="subtract")
snn_mem = snn_lif.init_leaky()

# TensorFlow reference
tf_weights = tf.constant(weights_fp32)
tf_mem = tf.zeros([NUM_HIDDEN], dtype=tf.float32)

print(f"Quantized INT8 Weights:\n{weights_int8}\n")
print("Step | Input Spikes | INT32 V_mem [N0, N1, N2] | Golden Spikes | snnTorch Spikes | TF Spikes")
print("-" * 88)

for t in range(TIME_STEPS):
    x_t = input_spikes[t]

    # 1. Golden Reference Model
    spk_hw, v_hw = golden_layer.step(x_t)

    # 2. snnTorch FP32
    cur_pt = torch.matmul(torch.tensor(weights_fp32), torch.tensor(x_t, dtype=torch.float32))
    spk_pt, snn_mem = snn_lif(cur_pt, snn_mem)
    spk_pt = spk_pt.detach().numpy().astype(np.uint8)

    # 3. TensorFlow FP32
    cur_tf = tf.linalg.matvec(tf_weights, tf.constant(x_t, dtype=tf.float32))
    tf_mem = 0.875 * tf_mem + cur_tf
    spk_tf = tf.cast(tf_mem >= V_TH_FLOAT, tf.uint8).numpy()
    tf_mem = tf_mem - tf.cast(spk_tf, tf.float32) * V_TH_FLOAT

    print(f" {t:02d}  |   {x_t}  |      {v_hw}       |    {spk_hw}    |     {spk_pt}     |  {spk_tf}")
