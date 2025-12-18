import cupy as cp

print(cp.cuda.runtime.getDeviceCount())  # Должно показать 1 (или больше)