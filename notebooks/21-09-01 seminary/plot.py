
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pymongo import MongoClient

def get_metric_matrix(collection, metric_name, dataset):
    x_values = collection.distinct('parameters.protected-promotion')
    y_values = collection.distinct('parameters.unprotected-demotion')
    result = np.zeros( (len(x_values), len(y_values)) )

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            item = collection.find_one({'parameters.protected-promotion': x, 'parameters.unprotected-demotion': y, 'parameters.dataset': dataset})
            metric = item['metrics'][metric_name]
            result[i,j] = metric

    return result, x_values, y_values

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:0.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              size=5)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def make_plot(data, name, x_labels, x_name, y_labels, y_name, dataset):
    fig, ax = plt.subplots()

    im, cbar = heatmap(data, x_labels, y_labels, ax=ax,
                       cmap="YlGn", cbarlabel=name)
    texts = annotate_heatmap(im, valfmt="{x:0.3f}")

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(name + '\n' + dataset)
    fig.tight_layout()
    plt.savefig('plots/dataset/' + name, dpi=300, facecolor='white', transparent=False)

client = MongoClient('localhost', 27017)
db = client['fairness']
collection = db['21_09_01_seminary']

metrics = ['ACC_balance',
'ACC_overall',
'ACC_relative_balance',
'FDR_balance',
'FDR_overall',
'FDR_relative_balance',
'FNR_balance',
'FNR_overall',
'FNR_relative_balance',
'FOR_balance',
'FOR_overall',
'FOR_relative_balance',
'FPR_balance',
'FPR_overall',
'FPR_relative_balance',
'MCC_balance',
'MCC_overall',
'MCC_relative_balance',
'NPV_balance',
'NPV_overall',
'NPV_relative_balance',
'Negatives_balance',
'Negatives_overall',
'Negatives_relative_balance',
'PPV_balance',
'PPV_overall',
'PPV_relative_balance',
'Positives_balance',
'Positives_overall',
'Positives_relative_balance',
'TNR_balance',
'TNR_overall',
'TNR_relative_balance',
'TPR_balance',
'TPR_overall',
'TPR_relative_balance']

datasets = ['german', 'income']
for dataset in datasets:
    for metric_name in metrics:
        data, protected_promotion, unprotected_demotion = get_metric_matrix(collection, metric_name, dataset)
        metric_name = metric_name.replace('_', ' ')
        make_plot(data, metric_name, protected_promotion, 'Protected promotion',
                  unprotected_demotion, 'Unprotected demotion', dataset)
