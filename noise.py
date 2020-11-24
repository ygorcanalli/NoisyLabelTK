#%%
from abc import ABCMeta, abstractmethod
import numpy as np
import warnings

# %%
class NoiseGenerator(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, labels, noise_rate):
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            warnings.warn("Multiclass data found, one-hot-encoding supposed, argmax applyed to one-hot-decode")
            self.labels = np.argmax(labels, axis=0)
        else:
            self.labels = labels

        self.classes, self.classes_index = np.unique(self.labels, return_index=True)
        self.n_classes = self.classes.shape[0]
        self.lenght = self.labels.shape[0]
        self.noise_rate = noise_rate

class TensorialNoiseGenerator(NoiseGenerator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, labels, noise_rate, transition_tensor=None):
        super().__init__(labels, noise_rate)
 
        # initialize with identity if tensor is not available
        if transition_tensor is None:
            self.transition_tensor = np.identity(self.n_classes)
        # check if each row sums zero
        elif (np.allclose(np.sum(transition_tensor, axis=0), np.ones(self.n_classes))):
            self.transition_tensor = transition_tensor
        else:
            raise Exception('Transition tensor must sum zero along first dimension')

class UniformNoiseGenerator(TensorialNoiseGenerator):
    
    def __init__(self, labels, noise_rate):

        # initialize transition matrix with identity
        super().__init__(labels, noise_rate)

        # fill probability of not change label
        self.transition_tensor = self.transition_tensor * (1 - self.noise_rate) 

        # equal probability of changing to each one lable. transition matrix row with sum = 1
        self.transition_tensor[self.transition_tensor == 0] = self.noise_rate/(self.n_classes - 1)



# %%
