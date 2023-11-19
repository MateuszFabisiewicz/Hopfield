from hopfield import HopfieldNetwork
from test_cases import TestCase
import os
import random
import multiprocessing
import numpy as np

dir = "Data"

test_cases = []
for _, _, files in os.walk(dir):
    for file in files:
        test_cases.append(TestCase(file, dir))

random.seed(10)

def func(test_case):
    print("Test case -- ", test_case.name)
    print()

    patterns_number = len(test_case.patterns)
    network_hebbian = HopfieldNetwork(test_case.shape[0] * test_case.shape[1])
    network_hebbian.train_hebbian(patterns=test_case.patterns)

    hebbian_acc = 0
    hebbian_async_acc = 0

    network_oja = HopfieldNetwork(test_case.shape[0] * test_case.shape[1])
    network_oja.train_oja(patterns=test_case.patterns)

    oja_acc = 0
    oja_async_acc = 0
    
    for pattern in test_case.patterns:
        random_pattern = np.random.choice([-1,1],size = len(pattern))
        retrived_pattern = network_oja.recall(random_pattern)
        oja_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_oja.recall_async(random_pattern)
        oja_async_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_hebbian.recall(random_pattern)
        hebbian_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_hebbian.recall_async(random_pattern)
        hebbian_async_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

    print()
    print(f"{test_case.name}: Hebbian synchronous accuracy in patterns: {hebbian_acc} ({round(100*hebbian_acc/patterns_number,2)}%)")

    print()
    print(f"{test_case.name}: Hebbian asynchronous accuracy in patterns: {hebbian_async_acc} ({round(100*hebbian_acc/patterns_number,2)}%)")
    
    print()
    print(f"{test_case.name}: Oja synchronous accuracy in patterns: {oja_acc} ({round(100*oja_acc/patterns_number,2)}%)")

    print()
    print(f"{test_case.name}: Oja asynchronous accuracy in patterns: {oja_async_acc} ({round(100*oja_acc/patterns_number,2)}%)")

if __name__ == '__main__':
    pool_obj = multiprocessing.Pool()
    ans = pool_obj.map(func,test_cases)
    print(ans)
    pool_obj.close()
    