import multiprocessing as mp
import time
import numpy as np



def worker_inner(input):
    time.sleep(np.random.uniform(0.1, 0.3))
    return input**2


def worker_outer(inputs):
    time.sleep(np.random.uniform(0.1, 0.3))
    fut = pool.map_async(worker_inner, inputs)
    return fut.get()


manager = mp.Manager()
pool = mp.Pool(12)


chunks = np.random.uniform(1, 100, (300,300))
results = pool.map_async(worker_outer, chunks)
print(results.get())
