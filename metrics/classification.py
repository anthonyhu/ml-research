import os
import json
import numpy as np


class AccuracyMetrics:
    def __init__(self, mode, tensorboard, session_name):
        self.mode = mode
        self.tensorboard = tensorboard
        self.session_name = session_name
        self.value = 0.0
        self.n_batch = 0

    def update(self, pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        pred = np.argmax(pred, axis=-1)
        self.value += (pred == target).mean()
        self.n_batch += 1

    def evaluate(self, global_step):
        if self.n_batch > 0:
            score = self.value / self.n_batch
        else:
            score = -float('inf')

        self.tensorboard.add_scalar(self.mode + '/accuracy', score, global_step)

        self.save_json(score)
        self.reset()
        return score

    def save_json(self, score):
        filename = os.path.join(self.session_name, self.mode + '_metrics.json')
        output = {'accuracy': score}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def reset(self):
        self.value = 0.0
        self.n_batch = 0
