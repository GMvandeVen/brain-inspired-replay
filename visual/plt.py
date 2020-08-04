import matplotlib
matplotlib.use('Agg')
# above 2 lines set the matplotlib backend to 'Agg', which
#  enables matplotlib-plots to also be generated if no X-server
#  is defined (e.g., when running in basic Docker-container)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.utils import make_grid
import numpy as np


def open_pdf(full_path):
    return PdfPages(full_path)


def plot_images_from_tensor(image_tensor, pdf=None, nrow=8, title=None, config=None):
    '''Plot images in [image_tensor] as a grid with [nrow] into [pdf].

    [image_tensor]      <tensor> [batch_size]x[channels]x[width]x[height]'''

    # -denormalize images if needed
    if config is not None and config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)
    # -create image-grad and plot
    image_grid = make_grid(image_tensor, nrow=nrow, pad_value=1)  # pad_value=0 gives black borders
    _ = plt.figure()
    plt.imshow(np.transpose(image_grid.numpy(), (1,2,0)))
    # -add title if provided
    if title:
        plt.title(title)
    # -save figure into pdf
    if pdf is not None:
        pdf.savefig()


def plot_scatter_groups(x, y, colors=None, ylabel=None, xlabel=None, title=None, top_title=None, names=None,
                        xlim=None, ylim=None, markers=None, figsize=None):
    '''Generate a figure containing a scatter-plot.'''

    # if needed, generate default group-names
    if names == None:
        n_points = len(y)
        names = ["group " + str(id) for id in range(n_points)]

    # make plot
    f, axarr = plt.subplots(1, 1, figsize=(12, 7) if figsize is None else figsize)
    for i,name in enumerate(names):
        # plot individual points
        axarr.scatter(x=x[i], y=y[i], color=None if (colors is None) else colors[i],
                      marker="o" if markers is None else markers[i], s=40, alpha=0.5)
        # plot group means
        axarr.scatter(x=np.mean(x[i]), y=np.mean(y[i]), color=None if (colors is None) else colors[i], label=name,
                      marker="*" if markers is None else markers[i], s=160)

    # finish layout
    # -set y/x-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    if xlim is not None:
        axarr.set_xlim(xlim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -add legend
    if names is not None:
        axarr.legend()

    # return the figure
    return f


def plot_scatter(x, y, colors=None, ylabel=None, xlabel=None, title=None, top_title=None, names=None,
                 xlim=None, ylim=None, markers=None):
    '''Generate a figure containing a scatter-plot.'''

    # if needed, generate default point-names
    if names == None:
        n_points = len(y)
        names = ["point " + str(id) for id in range(n_points)]

    # make plot
    f, axarr = plt.subplots(1, 1, figsize=(12, 7))
    for i,name in enumerate(names):
        axarr.scatter(x=x[i], y=y[i], color=None if (colors is None) else colors[i], label=name,
                      marker="*" if markers is None else markers[i], s=160)

    # finish layout
    # -set y/x-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    if xlim is not None:
        axarr.set_xlim(xlim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -add legend
    if names is not None:
        axarr.legend()

    # return the figure
    return f


def plot_bar(numbers, names=None, colors=None, ylabel=None, title=None, top_title=None, ylim=None, figsize=None,
             yerr=None):
    '''Generate a figure containing a bar-graph.'''

    # number of bars
    n_bars = len(numbers)

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)
    axarr.bar(x=range(n_bars), height=numbers, color=colors, yerr=yerr)

    # finish layout
    axarr.set_xticks(range(n_bars))
    if names is not None:
        axarr.set_xticklabels(names, rotation=-20)
        axarr.legend()
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)

    # return the figure
    return f


def plot_bars(number_list, names=None, colors=None, ylabel=None, title_list=None, top_title=None, ylim=None,
              figsize=None, yerr=None, vlines=None, alpha=None, dots=None,
              h_line=None, h_label=None, h_lines=None, h_colors=None, h_labels=None, h_errors=None):
    '''Generate a figure containing multiple bar-graphs.

    [number_list]   <list> with <lists> of numbers to plot in each sub-graph
    [names]         <list> (with <lists>) of names for axis
    [colors]        <list> (with <lists>) of colors'''

    # number of plots
    n_plots = len(number_list)

    # number of bars per plot
    n_bars = []
    for i in range(n_plots):
        n_bars.append(len(number_list[i]))

    # decide on scale y-axis
    maxY = np.max([np.max(list) for list in number_list])
    if dots is not None:
        maxY = np.max([np.max([np.max(list) for list in list_of_list]) for list_of_list in dots])
    y_max = maxY+0.07*maxY

    # make figure
    size = (16,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, n_plots, figsize=size)

    # make all plots
    for i in range(n_plots):
        axarr[i].bar(x=range(n_bars[i]), height=number_list[i], color=colors[i] if type(colors[0])==list else colors,
                     yerr=yerr[i] if yerr is not None else None, alpha=alpha[i] if type(alpha)==list else alpha)

        # finish layout for this plot
        if ylim is None:
            axarr[i].set_ylim(0, y_max)
        else:
            axarr[i].set_ylim(ylim)
        axarr[i].set_xticks(range(n_bars[i]))
        if names is not None:
            axarr[i].set_xticklabels(names[i] if type(names[0])==list else names, rotation=-20)
            axarr[i].legend()
        if i==0 and ylabel is not None:
            axarr[i].set_ylabel(ylabel)
        if title_list is not None:
            axarr[i].set_title(title_list[i])
        if vlines is not None:
            axarr[i].axvline(x=vlines[i] if type(vlines)==list else vlines, color="black", linewidth=0.7)
        if dots is not None:
            for j in range(len(dots[i])):
                n_dots = len(dots[i][j])
                axarr[i].plot([j]*n_dots + np.random.uniform(-0.18, 0.18, size=n_dots), dots[i][j], color="black",
                              marker='.', linewidth=0)

        # add single dashed horizontal line
        if h_line is not None:
            axarr[i].axhline(y=h_line, label=h_label, color="black", linestyle='dashed')

        # add (multiple) horizontal line(s), possibly with error-bars
        if h_lines is not None:
            for line_id, new_h_line in enumerate(h_lines):
                axarr[i].axhline(y=new_h_line, label=None if h_labels is None else h_labels[line_id],
                                 color=None if (h_colors is None) else h_colors[line_id])
                if h_errors is not None:
                    axarr[i].fill_between(axarr[i].get_xlim(),
                                          [new_h_line + h_errors[line_id], new_h_line + h_errors[line_id]],
                                          [new_h_line - h_errors[line_id], new_h_line - h_errors[line_id]],
                                          color=None if (h_colors is None) else h_colors[line_id], alpha=0.25)

        # add legend
        if (h_line is not None and h_label is not None) or (h_lines is not None and h_labels is not None):
            axarr[i].legend()

    # finish global layout
    if top_title is not None:
        f.suptitle(top_title)

    # return the figure
    return f


def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
               title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
               x_log=False, with_dots=False, linestyle='solid', h_line=None, h_label=None, h_error=None,
               h_lines=None, h_colors=None, h_labels=None, h_errors=None):
    '''Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)

    # add error-lines / shaded areas
    if list_with_errors is not None:
        for task_id, name in enumerate(line_names):
            if errors=="shaded":
                axarr.fill_between(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])),
                                   list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])),
                                   color=None if (colors is None) else colors[task_id], alpha=0.25)
            else:
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')
                axarr.plot(x_axes, list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])), label=None,
                           color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')

    # mean lines
    for task_id, name in enumerate(line_names):
        axarr.plot(x_axes, list_with_lines[task_id], label=name,
                   color=None if (colors is None) else colors[task_id],
                   linewidth=4, marker='o' if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[task_id])

    # add horizontal line
    if h_line is not None:
        axarr.axhline(y=h_line, label=h_label, color="grey")
        if h_error is not None:
            if errors == "shaded":
                axarr.fill_between([x_axes[0], x_axes[-1]],
                                   [h_line + h_error, h_line + h_error], [h_line - h_error, h_line - h_error],
                                   color="grey", alpha=0.25)
            else:
                axarr.axhline(y=h_line + h_error, label=None, color="grey", linewidth=1, linestyle='dashed')
                axarr.axhline(y=h_line - h_error, label=None, color="grey", linewidth=1, linestyle='dashed')

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for task_id, new_h_line in enumerate(h_lines):
            axarr.axhline(y=new_h_line, label=None if h_labels is None else h_labels[task_id],
                          color=None if (h_colors is None) else h_colors[task_id])
            if h_errors is not None:
                if errors == "shaded":
                    axarr.fill_between([x_axes[0], x_axes[-1]],
                                       [new_h_line + h_errors[task_id], new_h_line+h_errors[task_id]],
                                       [new_h_line - h_errors[task_id], new_h_line - h_errors[task_id]],
                                       color=None if (h_colors is None) else h_colors[task_id], alpha=0.25)
                else:
                    axarr.axhline(y=new_h_line+h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')
                    axarr.axhline(y=new_h_line-h_errors[task_id], label=None,
                                  color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                                  linestyle='dashed')

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if line_names is not None:
        axarr.legend()
    # -set x-axis to log-scale
    if x_log:
        axarr.set_xscale('log')

    # return the figure
    return f



def plot_lines_with_baselines(
        list_with_lines, x_axes=None, line_names=None, colors=None, title=None, title_top=None, xlabel=None,
        ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded", x_log=False, with_dots=False,
        linestyle='solid', h_lines=None, h_colors=None, h_labels=None, h_errors=None
):
    '''Generates a figure containing multiple lines, with a sideplot depicting the baselines (i.e., [h_lines]).

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12, 7) if figsize is None else figsize
    f, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=size)

    # add error-lines / shaded areas
    if list_with_errors is not None:
        for task_id, name in enumerate(line_names):
            if errors == "shaded":
                ax1.fill_between(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])),
                                 list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])),
                                 color=None if (colors is None) else colors[task_id], alpha=0.25)
            else:
                ax1.plot(x_axes, list(np.array(list_with_lines[task_id]) + np.array(list_with_errors[task_id])),
                         label=None,
                         color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')
                ax1.plot(x_axes, list(np.array(list_with_lines[task_id]) - np.array(list_with_errors[task_id])),
                         label=None,
                         color=None if (colors is None) else colors[task_id], linewidth=1, linestyle='dashed')

    # mean lines
    for task_id, name in enumerate(line_names):
        ax1.plot(x_axes, list_with_lines[task_id], label=name,
                 color=None if (colors is None) else colors[task_id],
                 linewidth=2, marker='o' if with_dots else None,
                 linestyle=linestyle if type(linestyle) == str else linestyle[task_id])

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for task_id, new_h_line in enumerate(h_lines):
            ax0.plot([task_id - 0.45, task_id + 0.45], [new_h_line, new_h_line],
                     label=None if h_labels is None else h_labels[task_id],
                     color=None if (h_colors is None) else h_colors[task_id])
            if h_errors is not None:
                if errors == "shaded":
                    ax0.fill_between([task_id - 0.45, task_id + 0.45],
                                     [new_h_line + h_errors[task_id], new_h_line + h_errors[task_id]],
                                     [new_h_line - h_errors[task_id], new_h_line - h_errors[task_id]],
                                     color=None if (h_colors is None) else h_colors[task_id], alpha=0.25)
                else:
                    ax0.plot([task_id - 0.45, task_id + 0.45],
                             [new_h_line + h_errors[task_id], new_h_line + h_errors[task_id]], label=None,
                             color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                             linestyle='dashed')
                    ax0.plot([task_id - 0.45, task_id + 0.45],
                             [new_h_line - h_errors[task_id], new_h_line - h_errors[task_id]], label=None,
                             color=None if (h_colors is None) else h_colors[task_id], linewidth=1,
                             linestyle='dashed')

    # finish layout
    ax0.set_xticks([])
    # -set y-axis
    if ylim is None:
        ylim0 = ax0.get_ylim()
        ylim1 = ax1.get_ylim()
        ylim = (min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1]))
    ax0.set_ylim(ylim)
    ax1.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    if ylabel is not None:
        ax0.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        ax1.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend(s)
    if line_names is not None:
        ax1.legend()
    if h_labels is not None:
        ax0.legend()
    # -set x-axis to log-scale
    if x_log:
        ax1.set_xscale('log')

    # return the figure
    return f



def plot_histogram(numbers, ylabel="frequency", xlabel=None, title=None, top_title=None, ylim=None, xlim=None,
                   figsize=None):
    '''Generate a figure containing a histogram.'''

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)
    axarr.hist(x=numbers)

    # finish layout
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    if title is not None:
        axarr.set_title(title)
    if top_title is not None:
        f.suptitle(top_title)
    # -set axes
    if ylim is not None:
        axarr.set_ylim(ylim)
    if xlim is not None:
        axarr.set_xlim(xlim)

    # return the figure
    return f



def plot_matrix(array, title=None, xlabel=None, ylabel=None, cmap=plt.cm.Blues, integers=False, figsize=None,
                xticklabels=None, yticklabels=None):
    '''Generate figure of 2D <np-array> plotted in color.'''

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)
    im = axarr.imshow(array, interpolation='nearest', cmap=cmap)

    # layout
    axarr.figure.colorbar(im)
    if title is not None:
        axarr.set_title(title)
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    axarr.set(xticks=np.arange(array.shape[1]), yticks=np.arange(array.shape[0]),
              xticklabels=np.arange(array.shape[1]) if xticklabels is None else xticklabels,
              yticklabels=np.arange(array.shape[0]) if yticklabels is None else yticklabels)

    # loop over data dimensions and create text annotations.
    fmt = 'd' if integers else '.2f'
    thresh = array.max() / 2.
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > 0:
                axarr.text(j, i, format(array[i, j], fmt), ha="center", va="center",
                           color="white" if array[i, j] > thresh else "black")

    # finalize layout
    f.tight_layout()

    # return the figure
    return f



def plot_pr_curves(precision_list, recall_list, names=None, colors=None,
                   figsize=None, with_dots=False, linestyle="solid", title=None, title_top=None, alpha=None):
    '''Generates a figure containing multiple groups of "Precision & Recall"-curves in one plot.

    :param precision_list:  <list> of all <lists> of precision-lines to plot (with each line being a <list> as well)
    :param receall_list:    <list> of all <lists> of precision-lines to plot (with each line being a <list> as well)
    :param names:           <list> containing the names of each group
    :param colors:          <list> containing the colors of each group
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>'''

    # defaults for "Precision & Recall"-curves
    ylim = xlim = [0, 1]
    xlabel = "Recall"
    ylabel = "Precision"

    # make plot
    size = (8, 8) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)

    # loop over all groups
    for group_id in range(len(precision_list)):
        new_group = True

        # loop over all lines
        n_lines = len(precision_list[group_id])
        for line_id in range(n_lines):
          axarr.plot(recall_list[group_id][line_id], precision_list[group_id][line_id], label=None,
                     color=colors[group_id] if colors is not None else "black", linewidth=2,
                     alpha=0.5*alpha if alpha is not None else 0.5, marker='o' if with_dots else None,
                     linestyle=linestyle if type(linestyle) == str else linestyle[group_id])
          if new_group:
              sum_recall = recall_list[group_id][line_id]
              sum_precision = precision_list[group_id][line_id]
              new_group = False
          else:
              sum_recall = [sum(x) for x in zip(sum_recall, recall_list[group_id][line_id])]
              sum_precision = [sum(x) for x in zip(sum_precision, precision_list[group_id][line_id])]

        # add mean group lines
        axarr.plot([rec/n_lines for rec in sum_recall], [pre/n_lines for pre in sum_precision],
                   label=names[group_id] if names is not None else None,
                   color=colors[group_id] if colors is not None else "black", linewidth=4,
                   marker='o' if with_dots else None,
                   linestyle=linestyle if type(linestyle) == str else linestyle[group_id],
                   alpha=alpha if alpha is not None else 1.)

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -set x-axis
    if xlim is not None:
        axarr.set_xlim(xlim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if names is not None:
        axarr.legend()

    return f