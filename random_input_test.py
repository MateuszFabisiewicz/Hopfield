from hopfield import HopfieldNetwork
from test_cases import TestCase
import os
import random
import multiprocessing
import numpy as np

dir = "Data"

def change_bits(pattern, num_bits):
    indices = random.sample(range(len(pattern)), num_bits)

    new_pattern = pattern.copy()
    for index in indices:
        new_pattern[index] *= -1
    return new_pattern

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
    hebbian_dim1_acc = 0
    hebbian_dim2_acc = 0
    hebbian_dim3_acc = 0
    hebbian_async_acc = 0
    hebbian_async_dim1_acc = 0
    hebbian_async_dim2_acc = 0
    hebbian_async_dim3_acc = 0

    network_oja = HopfieldNetwork(test_case.shape[0] * test_case.shape[1])
    network_oja.train_oja(patterns=test_case.patterns)

    oja_acc = 0
    oja_dim1_acc = 0
    oja_dim2_acc = 0
    oja_dim3_acc = 0
    oja_async_acc = 0
    oja_async_dim1_acc = 0
    oja_async_dim2_acc = 0
    oja_async_dim3_acc = 0

    
    iters = 3
    for pattern in test_case.patterns:
        #random_pattern = np.random.choice([-1,1],size = len(pattern))
        retrived_pattern = network_oja.recall(pattern)
        oja_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_oja.recall_async(pattern)
        oja_async_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_hebbian.recall(pattern)
        hebbian_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

        retrived_pattern = network_hebbian.recall_async(pattern)
        hebbian_async_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
        
        for _ in range(iters):
            pattern1 = change_bits(pattern,int(np.ceil(network_hebbian.num_neurons/100)))
            pattern2 = change_bits(pattern,int(np.ceil(3*network_hebbian.num_neurons/100)))
            pattern3 = change_bits(pattern,int(np.ceil(10*network_hebbian.num_neurons/100)))

            retrived_pattern = network_hebbian.recall(pattern1)
            hebbian_dim1_acc += (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_hebbian.recall_async(pattern1)
            hebbian_async_dim1_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            

            retrived_pattern = network_hebbian.recall(pattern2)
            hebbian_dim2_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_hebbian.recall_async(pattern2)
            hebbian_async_dim2_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            

            retrived_pattern = network_hebbian.recall(pattern3)
            hebbian_dim3_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_hebbian.recall_async(pattern3)
            hebbian_async_dim3_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            

            retrived_pattern = network_oja.recall(pattern1)
            oja_dim1_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_oja.recall_async(pattern1)
            oja_async_dim1_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            

            retrived_pattern = network_oja.recall(pattern2)
            oja_dim2_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_oja.recall_async(pattern2)
            oja_async_dim2_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            

            retrived_pattern = network_oja.recall(pattern3)
            oja_dim3_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)
            
            retrived_pattern = network_oja.recall_async(pattern3)
            oja_async_dim3_acc+= (retrived_pattern == pattern).sum()/float(retrived_pattern.size)

    hebbian_dim1_acc /= iters
    hebbian_dim2_acc /= iters
    hebbian_dim3_acc /= iters
    hebbian_async_dim1_acc /= iters
    hebbian_async_dim2_acc /= iters
    hebbian_async_dim3_acc /= iters
    oja_dim1_acc /= iters
    oja_dim2_acc /= iters
    oja_dim3_acc /= iters
    oja_async_dim1_acc /= iters
    oja_async_dim2_acc /= iters
    oja_async_dim3_acc /= iters

    print()
    print(f"{test_case.name}: Hebbian synchronous accuracy in patterns: {hebbian_acc} ({round(100*hebbian_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian synchronous accuracy in patterns when 1% malformed: {hebbian_dim1_acc} ({round(100*hebbian_dim1_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian synchronous accuracy in patterns when 3% bits malformed: {hebbian_dim2_acc} ({round(100*hebbian_dim2_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian synchronous accuracy in patterns when 10% bits malformed: {hebbian_dim3_acc} ({round(100*hebbian_dim3_acc/patterns_number,2)}%)")

    print()
    print(f"{test_case.name}: Hebbian asynchronous accuracy in patterns: {hebbian_async_acc} ({round(100*hebbian_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian asynchronous accuracy in patterns when 1% malformed: {hebbian_async_dim1_acc} ({round(100*hebbian_async_dim1_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian asynchronous accuracy in patterns when 3% bits malformed: {hebbian_async_dim2_acc} ({round(100*hebbian_async_dim2_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Hebbian asynchronous accuracy in patterns when 10% bits malformed: {hebbian_async_dim3_acc} ({round(100*hebbian_async_dim3_acc/patterns_number,2)}%)")
    
    print()
    print(f"{test_case.name}: Oja synchronous accuracy in patterns: {oja_acc} ({round(100*oja_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja synchronous accuracy in patterns when 1% malformed: {oja_dim1_acc} ({round(100*oja_dim1_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja synchronous accuracy in patterns when 3% bits malformed: {oja_dim2_acc} ({round(100*oja_dim2_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja synchronous accuracy in patterns when 10% bits malformed: {oja_dim3_acc} ({round(100*oja_dim3_acc/patterns_number,2)}%)")

    print()
    print(f"{test_case.name}: Oja asynchronous accuracy in patterns: {oja_async_acc} ({round(100*oja_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja asynchronous accuracy in patterns when 1% malformed: {oja_async_dim1_acc} ({round(100*oja_async_dim1_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja asynchronous accuracy in patterns when 3% bits malformed: {oja_async_dim2_acc} ({round(100*oja_async_dim2_acc/patterns_number,2)}%)")
    print(f"{test_case.name}: Oja asynchronous accuracy in patterns when 10% bits malformed: {oja_async_dim3_acc} ({round(100*oja_async_dim3_acc/patterns_number,2)}%)")

if __name__ == '__main__':
    pool_obj = multiprocessing.Pool()
    ans = pool_obj.map(func,test_cases)
    print(ans)
    pool_obj.close()
    