import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainers.trainer import Trainer
from data.cifar import CifarDataset
from models.cifar import CifarModel
from metrics.classification import AccuracyMetrics


class CifarTrainer(Trainer):
    def create_data(self):
        self.train_dataset = CifarDataset(mode='train')
        self.val_dataset = CifarDataset(mode='val')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                           num_workers=self.config.n_workers, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.batch_size,
                                         num_workers=self.config.n_workers, shuffle=False)

    def create_model(self):
        self.model = CifarModel()

    def create_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = AccuracyMetrics('train', self.tensorboard, self.session_name)
        self.val_metrics = AccuracyMetrics('val', self.tensorboard, self.session_name)

    def forward_model(self, batch):
        return self.model(batch['image'])

    def forward_loss(self, batch, output):
        return self.loss_fn(output, batch['label'])

    def visualise(self, batch, output, mode):
        self.tensorboard.add_images(mode + '/image', batch['image'], self.global_step)
