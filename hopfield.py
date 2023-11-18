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

    def train_oja(self, patterns):
        for pattern in patterns:
            if pattern.shape != (self.num_neurons,):
                raise ValueError("Pattern shape does not match network size.")
            
            weights = self.weights.copy()
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if i != j:
                        self.weights[i, j] += pattern[j] * (pattern[i] - weights[i, j] * pattern[j])
        self.weights /= len(patterns)

    def recall(self, pattern, max_iters=100):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        for _ in range(max_iters):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return (new_pattern, True)
            pattern = new_pattern
        
        return (pattern, False)
    
    def recall_async(self, pattern, max_iters=100):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        neuron_order = list(range(self.num_neurons))
        for _ in range(max_iters):
            new_pattern = pattern
            for neuron in neuron_order:
                new_pattern[neuron] = np.sign(np.dot(self.weights[neuron], new_pattern))
            
            if np.array_equal(new_pattern, pattern):
                return (new_pattern, True)
            pattern = new_pattern

        return (pattern, False)

    
    def plot_patterns_as_bitmap(self, patterns, image_size=(32, 32), save_path='patterns.bmp'):
        num_patterns = len(patterns)
        pattern_size = int(np.sqrt(self.num_neurons))
        cols = num_patterns
        rows = 1
        bitmap = np.ones((rows * pattern_size, cols * pattern_size), dtype=np.uint8) * 255
        
        for i in range(num_patterns):
            pattern = (patterns[i].reshape(pattern_size, pattern_size) + 1) * 127
            row = i // cols
            col = i % cols
            bitmap[row * pattern_size:(row + 1) * pattern_size, col * pattern_size:(col + 1) * pattern_size] = pattern
        
        plt.imsave(save_path, bitmap, cmap='gray')
        plt.show()

