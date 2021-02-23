from noisylabeltk.experiment import Experiment

T = [[0.9, 0.1],
     [0.2, 0.8]]
loss_args = [T]
loss_kwargs = {}
noise_args = [T]

parameters = {
    'batch_size': 64,
    'epochs': 10,
    'dataset': 'breast-cancer',
    'model': 'simple_mlp',
    'noise': 'pairwise',
    'noise-args': noise_args,
    'robust-method': 'forward',
    'loss-args': loss_args,
    'loss-kwargs': loss_kwargs
}

exp = Experiment(parameters)
exp.run()