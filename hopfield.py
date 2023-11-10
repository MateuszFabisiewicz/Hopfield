import numpy as np
import random
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train_hebbian_sync(self, patterns):
        for pattern in patterns:
            if pattern.shape != (self.num_neurons,):
                raise ValueError("Pattern shape does not match network size.")
            
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)

    def train_hebbian_async(self, patterns, num_epochs=10):
        for _ in range(num_epochs):
            for pattern in patterns:
                if pattern.shape != (self.num_neurons,):
                    raise ValueError("Pattern shape does not match network size.")
                
                neuron_order = list(range(self.num_neurons))
                random.shuffle(neuron_order)
                for i in neuron_order:
                    for j in range(self.num_neurons):
                        if i != j:
                            self.weights[i, j] += pattern[i] * pattern[j]

    def train_oja_sync(self, patterns, learning_rate=0.1):
        for pattern in patterns:
            if pattern.shape != (self.num_neurons,):
                raise ValueError("Pattern shape does not match network size.")
            
            for i in range(self.num_neurons):
                for j in range(self.num_neurons):
                    if i != j:
                        self.weights[i, j] += learning_rate * pattern[i] * (pattern[j] - self.weights[i, j] * pattern[i])

    def train_oja_async(self, patterns, learning_rate=0.1, num_epochs=10):
        for _ in range(num_epochs):
            for pattern in patterns:
                if pattern.shape != (self.num_neurons,):
                    raise ValueError("Pattern shape does not match network size.")
                
                neuron_order = list(range(self.num_neurons))
                random.shuffle(neuron_order)
                for i in neuron_order:
                    for j in range(self.num_neurons):
                        if i != j:
                            self.weights[i, j] += learning_rate * pattern[i] * (pattern[j] - self.weights[i, j] * pattern[i])

    def recall(self, pattern, max_iters=100):
        if pattern.shape != (self.num_neurons,):
            raise ValueError("Pattern shape does not match network size.")
        
        for _ in range(max_iters):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern
        
        return pattern
    
    def plot_patterns_as_bitmap(self, patterns, image_size=(32, 32), save_path='patterns.bmp'):
        num_patterns = len(patterns)
        pattern_size = int(np.sqrt(self.num_neurons))
        cols = int(np.sqrt(num_patterns))
        rows = num_patterns // cols
        bitmap = np.ones((rows * pattern_size, cols * pattern_size), dtype=np.uint8) * 255
        
        for i in range(num_patterns):
            pattern = (patterns[i].reshape(pattern_size, pattern_size) + 1) * 127
            row = i // cols
            col = i % cols
            bitmap[row * pattern_size:(row + 1) * pattern_size, col * pattern_size:(col + 1) * pattern_size] = pattern
        
        plt.imsave(save_path, bitmap, cmap='gray')
        plt.show()

# Example usage
if __name__ == "__main__":
    patterns = np.array([
        [1, 1, -1, -1],
        [-1, -1, 1, 1],
        [1, -1, -1, 1]
    ])

    network = HopfieldNetwork(num_neurons=4)

    # Train using synchronous Hebbian update
    network.train_hebbian_sync(patterns)

    test_pattern = np.array([1, 1, -1, 1])
    retrieved_pattern = network.recall(test_pattern)

    print("Test Pattern:", test_pattern)
    print("Retrieved Pattern (Synchronous Hebbian):", retrieved_pattern)

    # Reset the network
    network = HopfieldNetwork(num_neurons=4)

    # Train using asynchronous Hebbian update
    network.train_hebbian_async(patterns, num_epochs=100)

    test_pattern = np.array([1, 1, -1, 1])
    retrieved_pattern = network.recall(test_pattern)

    print("Retrieved Pattern (Asynchronous Hebbian):", retrieved_pattern)

    # Reset the network
    network = HopfieldNetwork(num_neurons=4)

    # Train using synchronous Oja update
    network.train_oja_sync(patterns, learning_rate=0.1)

    test_pattern = np.array([1, 1, -1, 1])
    retrieved_pattern = network.recall(test_pattern)

    print("Retrieved Pattern (Synchronous Oja):", retrieved_pattern)

    # Reset the network
    network = HopfieldNetwork(num_neurons=4)

    # Train using asynchronous Oja update
    network.train_oja_async(patterns, learning_rate=0.1, num_epochs=100)

    test_pattern = np.array([1, 1, -1, 1])
    retrieved_pattern = network.recall(test_pattern)

    print("Retrieved Pattern (Asynchronous Oja):", retrieved_pattern)

    # Save the training patterns as a bitmap image
    network.plot_patterns_as_bitmap(patterns, save_path='patterns.bmp')
