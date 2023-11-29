# -*- coding: utf-8 -*-

__all__ = ["factory", "runtime", "delay"]

import os
import time
import random
import functools
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import logging

log = logging.getLogger('SmoPa3D')

def factory(**default_opts):
    """
    Factory function to create decorators for tasks's run methods. Default options for the decorator
    function can be given in *default_opts*. The returned decorator can be used with or without
    actual invocation. Example:
    .. code-block:: python
        @factory(digits=2)
        def runtime(fn, opts, task, *args, **kwargs):
            t0 = time.time()
            try:
                return fn(task, *args, **kwargs)
            finally:
                t1 = time.time()
                diff = round(t1 - t0, opts["digits"])
                print("runtime:")
                print(diff)
        ...
        class MyTask():
            @runtime
            def run(self):
                ...
            # or
            @runtime(digits=3):
            def run(self):
                ...
    """
    def wrapper(decorator):
        @functools.wraps(decorator)
        def wrapper(fn=None, **opts):
            _opts = default_opts.copy()
            _opts.update(opts)

            def wrapper(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    return decorator(fn, _opts, *args, **kwargs)
                return wrapper

            return wrapper if fn is None else wrapper(fn)
        return wrapper
    return wrapper


@factory(digits=2)
def runtime(fn, opts, *args, **kwargs):
    """
    Decorator for inspecting a methods performance with a precision of *digits=2*.
    """
    t0 = time.time()
    try:
        return fn(*args, **kwargs)
    finally:
        t1 = time.time()
        if (t1-t0) < 2:
            diff = round(1000*(t1 - t0), opts["digits"])
            log.debug(f"{fn.__name__} runtime: {diff} ms")
        else:
            diff = round((t1 - t0), opts["digits"])
            log.debug(f"{fn.__name__} runtime: {diff} s")


@factory(t=5, stddev=0, pdf="gauss")
def delay(fn, opts, *args, **kwargs):
    """ delay(t=5, stddev=0., pdf="gauss")
    Wraps a bound method of a task and delays its execution by *t* seconds.
    """
    if opts["stddev"] <= 0:
        t = opts["t"]
    elif opts["pdf"] == "gauss":
        t = random.gauss(opts["t"], opts["stddev"])
    elif opts["pdf"] == "uniform":
        t = random.uniform(opts["t"], opts["stddev"])
    else:
        raise ValueError(
            "unknown delay decorator pdf '{}'".format(opts["pdf"]))

    time.sleep(t)

    return fn(*args, **kwargs)

class Tracker:

    def __init__(self) -> None:
        self.runtime:list[float] = []
        # self.memory:list[float] = []

    def plot_results(self, function_name:str, path:str='./__runtime_stats__'):
        os.makedirs(path, exist_ok=True)
        fig, axs = plt.subplots(1, 1, layout='constrained')

        # Plot runtime
        axs.plot(self.runtime)
        axs.set_ylabel('Runtime [ms]')
        axs.set_xlabel('Iterations')
        axs.set_title('Runtime')

        # # Plot memory
        # axs[1].plot(self.memory)
        # axs[1].set_ylabel('Memory [MiB]')
        # axs[1].set_xlabel('Iterations')
        # axs[1].set_title('Memory')

        print(f'Code statistics saved in {os.path.abspath(path)}')
        fig.savefig(f"{path}/{function_name}.png")
        

@factory()
def track(fn, opts, *args, **kwargs):
    """Keep track of the runtime of a function over time"""
    if f"{fn.__name__}_tracker" not in globals().keys():
        globals()[f"{fn.__name__}_tracker"] = Tracker()
    tracker:Tracker = globals()[f"{fn.__name__}_tracker"]
    t0 = time.time()
    try:
        # memory = max(memory_usage((fn, [*args], {**kwargs})))
        return fn(*args, **kwargs)
    finally:
        t1 = time.time()
        tracker.runtime.append(round((t1-t0)*1000, 1))
        # if memory is not None: tracker.memory.append(memory)

def save_tracking_stats():
    """Save the statistics in charts"""
    to_delete = []
    for varname, value in globals().items():
        if isinstance(value, Tracker):
            value.plot_results(function_name=varname)
            to_delete.append(varname)
    for varname in to_delete:
        del globals()[varname]