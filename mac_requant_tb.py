import numpy as np

# 1. NPU MAC Unit (INT8 -> INT32 Accumulator -> Requantized INT8 Output)
def npu_mac_requant(inputs_int8, weights_int8, shift=6):
    acc_int32 = np.sum(inputs_int8.astype(np.int32) * weights_int8.astype(np.int32), dtype=np.int32)
    out_int8 = np.clip(acc_int32 >> shift, -128, 127).astype(np.int8)
    return acc_int32, out_int8

# 2. Testbench Vectors: (Inputs, Expected_INT32_Acc, Expected_INT8_Out)
weights = np.array([64, 32, -48, 16], dtype=np.int8)

test_cases = [
    (np.array([10, 20, -15, 30], dtype=np.int8),     2480,   38),
    (np.array([-20, -10, 25, -5], dtype=np.int8),   -2880,  -45),
    (np.array([100, 100, -100, 50], dtype=np.int8), 15200,  127),
    (np.array([-100, -100, 100, -50], dtype=np.int8), -15200, -128),
]

# 3. Run Verification
for i, (x_in, exp_acc, exp_out) in enumerate(test_cases):
    acc, out = npu_mac_requant(x_in, weights, shift=6)
    status = "PASS" if (acc == exp_acc and out == exp_out and out.dtype == np.int8) else "FAIL"
    print(f"Test {i} [{status}] | INT32 Acc: {acc:6d} | INT8 Out: {out:4d} (Expected: {exp_out:4d})")
