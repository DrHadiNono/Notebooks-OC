from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

# Librairies pour la visualisation de graphiques
sns.set()  # Définir le style par défaut pour les graphiques
sns.set_palette("tab10")
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set_style(rc={'axes.facecolor': '#EBEBEB'})


def setup_plot(title=None, xlabel=None, ylabel=None, plot_fig_args={}):
    if len(plot_fig_args) < 3:
        plot_fig_args = {
            'width': 10,
            'height': 5,
            'dpi': 100
        }
    fig, ax = plt.subplots(
        figsize=(plot_fig_args['width'], plot_fig_args['height']))
    fig.set_dpi(plot_fig_args['dpi'])

    if title is not None:
        fig.suptitle(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylabel is not None:
        ax.set_xlabel(xlabel)

    return ax


def lineplot(series, title=None, xlabel=None, ylabel=None, label='', marker='-', plot_fig_args={}, return_plot=False, ax=None, ticks=None):
    if ax is None:
        ax = setup_plot(title, xlabel, ylabel, plot_fig_args)
    else:
        if title is not None and title != '':
            ax.set_title(title)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if ylabel is not None:
            ax.set_xlabel(xlabel)

    if isinstance(series, list) and len(series) == 2:
        ax.plot(series[0], series[1], marker, label=label)
    else:
        ax.plot(series, marker, label=label)

    if label != '':
        ax.legend(loc='best')
    if ticks is not None:
        ticks = series.index[range(0, len(series), len(series)//10)]
        ax.set_xticks(ticks)

    if return_plot:
        return ax


def boxplot(series, index=None, title=None, xlabel=None, ylabel=None, plot_fig_args={}):
    ax = setup_plot(title, xlabel, ylabel, plot_fig_args)

    if index is None:
        sns.boxplot(series, ax=ax)
    else:
        sns.boxplot(y=series, x=index, ax=ax)
