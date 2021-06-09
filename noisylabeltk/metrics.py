import numpy as np

def evaluate_discrimination(dataset_loader, model):
    dataset = dataset_loader.datasets[dataset_loader.name]()

    pred = model.predict(dataset['test']['features'])
    pred = np.argmax(pred, axis=1).reshape(pred.shape[0], 1)

    sensitive = (dataset['test']['sensitive'] == dataset['sensitive_positive_value']).to_numpy()
    accepted_positive = (pred & sensitive).sum()
    accepted_negative = (pred & ~sensitive).sum()

    total_positive = sensitive.sum()
    total_negative = (~sensitive).sum()

    return accepted_positive/total_positive - accepted_negative/total_negative