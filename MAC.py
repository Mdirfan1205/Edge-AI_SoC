
import numpy as np

def mac_int8(a, b, acc=0):

    # Convert inputs to signed INT8
    a = np.int8(a)
    b = np.int8(b)

    # Multiply using INT32
    product = np.int32(a) * np.int32(b)

    # Accumulate using INT32
    result = np.int32(acc) + product

    return np.int32(result)


# Test MAC operation

a = 10
b = 5
acc = 20

output = mac_int8(a, b, acc)

print("Input A:", a)
print("Input B:", b)
print("Accumulator:", acc)
print("MAC Output:", output)
