import numpy as np
import time

size = 10**7
a = np.random.rand(size)
b = np.random.rand(size)

start = time.time()
c = a + b
print(f"Time taken (Python CPU): {time.time() - start:.4f} seconds")

