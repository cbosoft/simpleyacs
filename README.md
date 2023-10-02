# Simple YACS Experiments

A simple example of using yacs to build config files for ML experiments.

This example task is for classifying an image based on whether it is light or dark. Dead easy, I know.

We have a resnet model for performing this task, and it has some hyperparameters.

We train the model, this has more hyperparameters.

We set up all the hyperparameters in the config file, and then we have an easy way to repeat an experiment.

## An example config file

Check out [`simpleyacs/config.py`](simpleyacs/config.py) to see the possible config options. This is a master list of all the hyperparameters.

Let's set up an experiment. I want to try a ResNet18, I want it to run for 10 epochs, I want to use a batch size of 10. The config file has a format as follows:

```yaml
# experiments/exp_test.yaml
action: train
training:
  batch_size: 10
  n_epochs: 10
model:
  n: 18
```

Then we run the experiment using the `run.py` script:

```bash
python run.py experiments/exp_test.yaml
```

The code will create an output directory, it will copy the config file there and it will also store results there.

For a less brief explanation, see [here](https://cmjb.tech/blog/2023/09/30/exp-config-files/).
