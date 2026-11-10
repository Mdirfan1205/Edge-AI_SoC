import numpy as np

# Fixed-point scale
SCALE = 32

# Original floating-point values
values = np.array([0.375, 0.8, 1.0, -0.5, 3.0])

# Convert to signed INT8
quantized = np.clip(
    np.round(values * SCALE),
    -128,
    127
).astype(np.int8)

# Convert back to approximate floating-point values
recovered = quantized.astype(np.float32) / SCALE

print("Original values:", values)
print("INT8 values:", quantized)
print("Recovered values:", recovered)
