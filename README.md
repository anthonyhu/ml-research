# ML research

This repository is the machine learning codebase as detailed in the following 
[guide](https://anthonyhu.github.io/research-workflow), which contains good practices 
for a machine learning researcher to structure day-to-day work.

## Quick start
* Clone the repo and create a [conda environment](https://anthonyhu.github.io/python-environment) with `conda env create`.
* Run training with `python run_training.py --config experiments/cifar.yml`.

For a new project, create a new trainer class in the `trainers` folder and implement the abstract methods of the general
`Trainer` class. See `trainers/trainer_cifar.py` for a detailed example.
