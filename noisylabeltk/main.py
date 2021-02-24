from noisylabeltk.experiment import Experiment

batch_size_list = [32]
epochs_list = [10]
dataset_list = ['german']
model_list = ['simple-mlp']

transition_matrix_list = []

transition_matrix_list.append([[0.9, 0.1],
                               [0.1, 0.9]])
transition_matrix_list.append([[0.8, 0.2],
                               [0.1, 0.9]])
transition_matrix_list.append([[0.9, 0.1],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.8, 0.2],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.7, 0.3],
                               [0.2, 0.8]])
transition_matrix_list.append([[0.2, 0.8],
                               [0.3, 0.7]])
transition_matrix_list.append([[0.7, 0.3],
                               [0.3, 0.7]])

robust_method_list = [
    # (robust-method,  (loss-args, loss-kwargs))
    ('none', ([], {})),
    ('boot-soft', ([], {})),
    ('boot-hard', ([], {}))
]

for transition_matrix in transition_matrix_list:
    robust_method_list.append( ('forward', ( [transition_matrix] , {})) )
    robust_method_list.append( ('backward', ( [transition_matrix] , {})) )

noise_list = [
    # (noise, noise-args)
    ('none', [])
]

noise_rate_list = [0.1, 0.2, 0.3]
for noise_rate in noise_rate_list:
    noise_list.append( ('uniform', [noise_rate]) )

for transition_matrix in transition_matrix_list:
    noise_list.append( ('pairwise', [transition_matrix]) )


for batch_size in batch_size_list:
    for epochs in epochs_list:
        for dataset in dataset_list:
            for model in model_list:
                for noise, noise_args in noise_list:
                    for robust_method, (loss_args, loss_kwargs) in robust_method_list:
                        parameters = {
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'dataset': dataset,
                            'model': model,
                            'noise': noise,
                            'noise-args': noise_args,
                            'robust-method': robust_method,
                            'loss-args': loss_args,
                            'loss-kwargs': loss_kwargs
                        }

                        exp = Experiment(parameters)
                        exp.run()
