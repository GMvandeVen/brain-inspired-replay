import numpy as np
from visdom import Visdom


_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_images(tensor, title, win=None, env='main', w=400, h=400, nrow=8):
    '''Plot images contained in 4D-tensor [X] to visdom-server.'''
    options = dict(title=title, width=w, height=h)
    win = title if win is None else win
    # if name in _WINDOW_CASH:
    #     _vis(env).close(win=_WINDOW_CASH.get(name))
    _WINDOW_CASH[win] = _vis(env).images(tensor, win=_WINDOW_CASH.get(win), nrow=nrow, opts=options)


def scatter_plot(X, title, colors=None, env='main', win=None,  w=400, h=400):
    '''Plot scatter-diagram of entries contained in 2D-tensor [X] to visdom-server.'''
    options = dict(title=title, width=w, height=h)
    win = title if win is None else win
    _WINDOW_CASH[win] = _vis(env).scatter(X, win=_WINDOW_CASH.get(win), Y=colors, opts=options)


def visualize_hist(X, title, win=None, env='main', w=400, h=400):
    '''Plot histogram of entries contained in 1D-tensor [X] to visdom-server.'''
    options = dict(title=title, width=w, height=h)
    win = title if win is None else win
    _WINDOW_CASH[win] = _vis(env).histogram(X, win=_WINDOW_CASH.get(win), opts=options)


def visualize_scalars(scalars, names, iteration, title, win=None, env='main', ylabel=None):
    assert len(scalars) == len(names)
    ylabel = title if (ylabel is None) else ylabel

    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu().numpy() if (hasattr(s, 'cpu') and hasattr(s.cpu(), 'numpy')) else np.array([s]) for s in scalars]
    num = len(scalars)

    options = dict(
        fillarea=False, legend=names, width=400, height=400,
        xlabel='Iterations', ylabel=ylabel, title=title,
        marginleft=30, marginright=30, marginbottom=80, margintop=30,
    )

    X = ( np.column_stack(np.array([iteration] * num)) if (num>1) else np.array([iteration] * num) )
    Y = np.column_stack(scalars) if (num>1) else scalars[0]

    win = title if win is None else win
    if win in _WINDOW_CASH:
        _vis(env).line(X=X, Y=Y, win=_WINDOW_CASH[win], opts=options, update='append')
    else:
        _WINDOW_CASH[win] = _vis(env).line(X=X, Y=Y, opts=options)
