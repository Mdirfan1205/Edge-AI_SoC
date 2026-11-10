
import numpy as np
import matplotlib.pyplot as plt

# LIF neuron parameters
beta = 0.8
threshold = 1.0
reset_voltage = 0.0

# Input current
inputs = np.array([0.4, 0.4, 0.4, 0.4, 0.4])

# Initial membrane voltage
voltage = 0.0

# Store results
voltages = []
spikes = []

# Simulate LIF neuron
for current in inputs:

    # Integrate and leak
    voltage = beta * voltage + current

    # Threshold check
    if voltage >= threshold:
        spike = 1
        voltage = reset_voltage
    else:
        spike = 0

    # Store results
    voltages.append(voltage)
    spikes.append(spike)

print("Membrane Voltage:", voltages)
print("Spike Output:", spikes)

# Plot results
plt.plot(voltages, marker="o", label="Voltage")
plt.axhline(threshold, linestyle="--", label="Threshold")
plt.xlabel("Time Step")
plt.ylabel("Membrane Voltage")
plt.title("LIF Neuron Simulation")
plt.legend()
plt.grid()
plt.show()
