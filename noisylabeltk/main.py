from noisylabeltk.experiment import Experiment


parameters = {
    'batch_size': 64,
    'epochs': 10,
    'dataset': 'breast-cancer',
    'model': 'simple_mlp',
    'noise': 'pairwise',
    'robust-method': 'none',
    'transition-matrix': [[0.9, 0.1],
                          [0.2, 0.8]]
}

exp = Experiment(parameters)
exp.run()