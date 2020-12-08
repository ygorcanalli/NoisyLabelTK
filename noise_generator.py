#%%
from abc import ABCMeta, abstractmethod
import numpy as np

# %%
class LabelNoiseGenerator(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, labels):
        if len(labels.shape) != 2 or labels.shape[1] < 2:
            raise Exception("Labels must be one-hot-encoded")

        self.labels = labels
        self.n_classes = self.labels.shape[1]
        self.classes = np.arange(self.n_classes)
        self.length = self.labels.shape[0]

    def check_noise_rate(self, noisy_labels):
        if noisy_labels.shape != (self.length, self.n_classes):
            raise Exception("Noisy labels must have same shape than original")
        accum = 0
        for i in range(n):
            accum += np.allclose(self.labels[i], noisy_labels[i])
        return 1 - accum/self.length
        

class TensorialLabelNoiseGenerator(LabelNoiseGenerator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, labels, transition_tensor=None):
        super().__init__(labels)
 
        # initialize with identity if tensor is not available
        if transition_tensor is None:
            self.transition_tensor = np.identity(self.n_classes)
        # check if each row sums zero
        elif (np.allclose(np.sum(transition_tensor, axis=1), np.ones(self.n_classes))):
            self.transition_tensor = transition_tensor
        else:
            raise Exception('Transition tensor must sum zero along first dimension')

    def generate_noisy_labels(self):
        noisy_labels = np.zeros( (self.length,self.n_classes) )
        probabilities = np.dot(self.labels, self.transition_tensor)
        for i in range(self.length):
            new_label = np.random.choice(self.classes, p=probabilities[i], replace=False)
            noisy_labels[i][new_label] = 1

        return noisy_labels

class UniformLabelNoiseGenerator(TensorialLabelNoiseGenerator):
    
    def __init__(self, labels, noise_rate):

        # initialize transition matrix with identity
        super().__init__(labels)
        self.noise_rate = noise_rate

        # fill probability of not change label
        self.transition_tensor = self.transition_tensor * (1 - self.noise_rate) 

        # equal probability of changing to each one lable. transition matrix row with sum = 1
        self.transition_tensor[self.transition_tensor == 0] = self.noise_rate/(self.n_classes - 1)

class PairwiseLabelNoiseGenerator(TensorialLabelNoiseGenerator):

    def __init__(self, labels, transition_matrix):
        if len(transition_matrix.shape) != 2:
            raise Exception("Transition matrix must to be a 2D array")
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise Exception("Transition matrix must to be square")
        super().__init__(labels, transition_matrix)