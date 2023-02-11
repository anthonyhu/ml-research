import os
import sys
from argparse import Namespace

import git


class Config(Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)


class Logger(object):
    """Save terminal outputs to log file, and continue to print on the terminal."""
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log = open(log_filename, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        pass

    def close(self):
        self.log.flush()
        os.fsync(self.log.fileno())
        self.log.close()


def get_git_hash():
    repository = git.Repo()
    git_hash = repository.head.object.hexsha
    return git_hash


def format_time(s):
    """Convert time in seconds to time in hours, minutes and seconds."""
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}h{m:02d}m{s:02d}s'


def print_model_spec(model, name=''):
    n_parameters = count_n_parameters(model)
    n_trainable_parameters = count_n_parameters(model, only_trainable=True)
    print(f'Model {name}: {n_parameters:.2f}M parameters of which {n_trainable_parameters:.2f}M are trainable.\n')


def count_n_parameters(model, only_trainable=False):
    if only_trainable:
        n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        n_parameters = sum([p.numel() for p in model.parameters()])
    return n_parameters / 10**6
