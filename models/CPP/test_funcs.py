import lvssa
import numpy as np
import matplotlib.pyplot as plt
import time

def test_add():
    c = np.ones((10,2))
    d = np.ones((10,2))
    d = d*2
    e = int(3)
    lvssa.LV(c,d,e)
    print(c)

if __name__ == '__main__':

    start = time.time()
    test_add()
    end = time.time()
    print(end - start)
