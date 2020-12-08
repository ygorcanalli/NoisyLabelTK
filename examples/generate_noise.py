# %%
from sklearn.preprocessing import OneHotEncoder
from .. import PairwiseLabelNoiseGenerator, UniformLabelNoiseGenerator
import numpy as np

n = 10000
m = 3

y = np.random.randint(0,m, (n,1) )

enc = OneHotEncoder()
y_encode = enc.fit_transform(y).toarray()

#%%
T = np.array([[0.7,0.2,0.1],
              [0.2,0.6,0.2],
              [0.05,0.05,0.9]])
gen = PairwiseLabelNoiseGenerator(y_encode, T)
noisy_labels = gen.generate_noisy_labels()
gen.check_noise_rate(noisy_labels)

#%%
y = np.random.randint(0,m, (n,1) )
y_encode = enc.fit_transform(y).toarray()
T = np.identity(m)
gen = PairwiseLabelNoiseGenerator(y_encode, T)
noisy_labels = gen.generate_noisy_labels()
gen.check_noise_rate(noisy_labels)
#


# %%
