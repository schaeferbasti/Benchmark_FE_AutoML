import time

from joblib import delayed, Parallel

# import Parallel

class SharedMem:
    def __init__(self):
        self._sharedCounter = 0

    def updater(self):
        time.sleep(3)
        self._sharedCounter+=1

    def getCounter(self):
        return self._sharedCounter

def TestSharedMem():
    sm = SharedMem()
    res = Parallel(n_jobs=-1, require='sharedmem')(delayed(sm.updater)() for i in range(10))
    print(res)
    print(sm._sharedCounter)

def awaiter():
    time.sleep(5)

def exp(a,b):
    time.sleep(5)
    return a**b

def main():
    # ParallelForEach(fibo, [[1],[2],[3],[4]])
    # ParallelForEach(awaiter, [[],[],[],[],[],[],[],[]])
    # print(Parallel.ParallelForEach(exp, [[i, i+1] for i in range(1,9)]))
    start = time.time()
    TestSharedMem()
    delta = time.time() - start
    print(delta)
if __name__=='__main__':
    main()