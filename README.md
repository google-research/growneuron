# GradMax: Growing Neural Networks using Gradient Information

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/growneuron/blob/main/Student_Teacher.ipynb)

Code for reproducing our results in the GradMax paper [[arxiv.org/abs/2201.05125](https://arxiv.org/abs/2201.05125)].

<img src="https://github.com/google-research/growneuron/blob/main/imgs/gradmax.png" alt="GradMax" width="80%" align="center">


## Setup
First clone this repo.
```bash
git clone https://github.com/google-research/growneuron.git
cd growneuron
```

Following script creates a virtual environment and
installs the necessary libraries. Finally, it runs few tests.
```bash
bash run.sh
```

We need to activate the virtual environment before running an experiment and
download the dataset. Is data-set is already downloaded the path can be passed
with `--data_dir` argument.
```bash
python growneuron/cifar/main.py --output_dir=/tmp/cifar --download_data
```

## Running GradMax
Following command would start a training with WRN-28-0.25x and grow it into
WRN-28-1. Growth is done every 2500 step starting from
iteration 10000 at all convolutional layers at once.
```bash
rm -rf /tmp/cifar
python growneuron/cifar/main.py --output_dir=/tmp/cifar \
--config=growneuron/cifar/configs/grow_all_at_once.py \
--config.grow_type=add_gradmax
```

## Other Experiments
- Baselines for WRN-28 and VGG11 can be ran using the corresponding configs in
`growneuron/cifar/configs/`.
- Set `--config.grow_type` argument to `add_gradmax`, `add_firefly`, `add_gradmax_opt` or
`add_random` to grow using different strategies.
- Use `--config.is_outgoing_zero` to run experiments where outgoing weights
are set to zero.
- Use `--config.model.normalization_type=batchnorm` to run experiments with
batch normalization layers.



## Disclaimer
This is not an officially supported Google product.
