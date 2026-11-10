
# Fixed-point LIF neuron simulation

SCALE = 32

# Quantized parameters
beta_q = 26
input_q = 13
threshold_q = 32
reset_q = 0

# Input signals
inputs = [input_q, input_q, input_q, input_q, input_q]

# Initial voltage
voltage_q = 0

voltages = []
spikes = []

# Round to nearest, ties away from zero
def round_divide(numerator, denominator):
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)

# LIF simulation
for current in inputs:

    # Leakage and integration
    voltage_q = round_divide(
        beta_q * voltage_q, SCALE
    ) + current

    # Threshold check
    if voltage_q >= threshold_q:
        spike = 1
        voltage_q = reset_q
    else:
        spike = 0

    voltages.append(voltage_q)
    spikes.append(spike)

# Display results
print("Fixed-point voltage:", voltages)
print("Recovered voltage:",
      [v / SCALE for v in voltages])
print("Spike output:", spikes)
