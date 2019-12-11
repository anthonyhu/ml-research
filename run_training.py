import argparse
from trainers.trainer_cifar import CifarTrainer

if __name__ == '__main__':
    options = argparse.ArgumentParser(description='Model config/restore path.')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options.add_argument('--restore', type=str, default='', help='Path of the model to restore (weights, optimiser)')
    options = options.parse_args()

    trainer = CifarTrainer(options)
    trainer.train()
