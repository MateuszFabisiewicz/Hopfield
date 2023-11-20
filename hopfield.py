import numpy as np
import random
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train_hebbian(self, patterns):
        for pattern in patterns:
            if pattern.shape != (self.num_neurons,):
                raise ValueError("Pattern shape does not match network size.")
            
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)
        self.weights = self.weights / len(patterns)

    def train_oja(self, patterns, iterations = 10):

        for i in range(iterations):
            weights = self.weights.copy()
            self.weights = np.zeros((self.num_neurons, self.num_neurons))
            for pattern in patterns:
                if pattern.shape != (self.num_neurons,):
                    raise ValueError("Pattern shape does not match network size.")
                
                for i in range(self.num_neurons):
                    for j in range(self.num_neurons):
                        if i != j:
                            self.weights[i, j] += pattern[j] * (pattern[i] - weights[i, j] * pattern[j])
            self.weights /= len(patterns)

    def recall(self, pattern, max_iters=100, show_details = False, plotShape = None):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        for _ in range(max_iters):
            weighted_sum = np.dot(self.weights, pattern)
            new_pattern = np.sign(weighted_sum) + (weighted_sum==0) 

            if(plotShape):
                self.plot_patterns_as_bitmap(new_pattern, plotShape)

            if(show_details):
                print(new_pattern)

            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern
        
        return pattern
    
    def is_stable(self, pattern):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        weighted_sum = np.dot(self.weights, pattern)
        new_pattern = np.sign(weighted_sum) + (weighted_sum==0) 
        if np.array_equal(new_pattern, pattern):
            return True
        return False
    
    def recall_async(self, pattern, max_iters=100, show_details=False, plotShape = None):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        neuron_order = list(range(self.num_neurons))
        for _ in range(max_iters):
            new_pattern = pattern.copy()
            for neuron in neuron_order:
                weighted_sum = np.dot(self.weights[neuron], new_pattern)
                new_pattern[neuron] = np.sign(weighted_sum) + (weighted_sum == 0)
                
            if(plotShape):
                self.plot_patterns_as_bitmap(new_pattern, plotShape)
            if(show_details):
                print(new_pattern)
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern

        return pattern
    
    def is_stable_async(self, pattern):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        neuron_order = list(range(self.num_neurons))
        new_pattern = pattern.copy()
        for neuron in neuron_order:
            weighted_sum = np.dot(self.weights[neuron], new_pattern)
            new_pattern[neuron] = np.sign(weighted_sum) + (weighted_sum == 0)

        if np.array_equal(new_pattern, pattern):
            return True
        return False

    
    def plot_patterns_as_bitmap(self, pattern, shape):
        fig = plt.figure(figsize=(shape[0]/5,shape[1]/5))
        pattern_array = np.array(pattern)
        pattern_array = pattern_array.reshape(shape)
        bitmap = np.where(pattern_array == 1, 255, 0)
        plt.axis("off")
        plt.imshow(bitmap, cmap='gray', interpolation='nearest', aspect="auto")

