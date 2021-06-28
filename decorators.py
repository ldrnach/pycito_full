"""
Useful function decorators for robotics and optimization problems

Luke Drnach
June 18, 2021
"""
import timeit
import functools
from matplotlib import pyplot as plt

#TODO: Figure out why showable_fig doesn't keep the plots open

def timer(func):
    """Print the runtime of the decorated function. The decorator also records the time in the total_time attribute"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        wrapper_timer.total_time = 0.
        start = timeit.default_timer()
        value = func(*args, **kwargs)
        stop = timeit.default_timer()
        wrapper_timer.total_time = stop - start
        print(f"Finished {func.__name__!r} in {wrapper_timer.total_time:.4f} seconds")
        return value
    return wrapper_timer

def saveable_fig(func):
    """Save a figure from the output"""
    @functools.wraps(func)
    def wrapper_saveable_fig(*args, **kwargs):
        fig, axs = func(*args, **kwargs)
        # Check kwargs for a savename and save the figure
        if 'savename' in kwargs and kwargs['savename'] is not None:
            fig.savefig(kwargs['savename'], dpi=fig.dpi)
        return fig, axs
    return wrapper_saveable_fig

def showable_fig(func):
    """Show a figure based on a show argument"""
    @functools.wraps(func)
    def wrapper_showable_fig(*args, **kwargs):
        fig, axs = func(*args, **kwargs)
        # Check kwargs for a show key and show the figure
        if 'show' in kwargs and kwargs['show']:
            plt.show(block=True)
        return fig, axs
    return wrapper_showable_fig

# @saveable_fig
# @showable_fig
# def testfig(show=False, savename=None):
#     fig, axs = plt.subplots(1,1)
#     axs.plot(0, 0)
#     return fig, axs

# if __name__ == '__main__':
#     testfig(show=True, savename=None)
