# Compare floating-point and fixed-point LIF neurons

# -------------------------------
# FLOATING-POINT LIF
# -------------------------------

beta = 0.8
input_current = 0.4
threshold = 1.0

float_voltage = 0.0

float_voltages = []
float_spikes = []

for _ in range(5):

    float_voltage = beta * float_voltage + input_current

    if float_voltage >= threshold:
        spike = 1
        float_voltage = 0.0
    else:
        spike = 0

    float_voltages.append(round(float_voltage, 4))
    float_spikes.append(spike)


# -------------------------------
# FIXED-POINT LIF
# -------------------------------

SCALE = 32

beta_q = 26
input_q = 13
threshold_q = 32
reset_q = 0

fixed_voltage = 0

fixed_voltages = []
fixed_spikes = []


def round_divide(numerator, denominator):
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


for _ in range(5):

    fixed_voltage = (
        round_divide(beta_q * fixed_voltage, SCALE)
        + input_q
    )

    if fixed_voltage >= threshold_q:
        spike = 1
        fixed_voltage = reset_q
    else:
        spike = 0

    fixed_voltages.append(fixed_voltage / SCALE)
    fixed_spikes.append(spike)


# -------------------------------
# DISPLAY RESULTS
# -------------------------------

print("FLOATING-POINT RESULTS")
print("Voltages:", float_voltages)
print("Spikes:  ", float_spikes)

print("\nFIXED-POINT RESULTS")
print("Voltages:", fixed_voltages)
print("Spikes:  ", fixed_spikes)