# coding=utf-8
# Copyright 2021 GradMax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default configs for baseline cifar-10 training."""
import ml_collections


def get_config():
  """Builds and returns config."""
  config = ml_collections.ConfigDict()

  config.optimizer = ml_collections.ConfigDict()
  # Base learning rate when total batch size is 128. It is scaled by the ratio
  # of the total batch size to 128.
  config.optimizer.base_learning_rate = 0.05
  # One of 'step', 'cosine'
  config.optimizer.decay_type = 'step'
  config.optimizer.nesterov = False
  # Amount to decay learning rate.
  config.optimizer.lr_decay_ratio = 0.1
  # Epochs to decay learning rate by.
  config.optimizer.lr_decay_epochs = [0.3, 0.6, 0.8]
  # Number of epochs for a linear warmup to the initial learning rate. Use 0 to'
  # do no warmup.
  config.optimizer.lr_warmup_epochs = 5
  # Optimizer momentum.
  config.optimizer.momentum = 0.9
  # Following is empty for the baselines and used by the growing algorithms.
  config.updater = ml_collections.ConfigDict()
  config.updater.carry_optimizer = False
  config.is_outgoing_zero = False
  config.scale_epochs = False

  config.model = ml_collections.ConfigDict()
  # L2 regularization coefficient.
  config.model.l2_coef = 1e-4
  config.model.width_multiplier = 0.25
  config.model.normalization_type = 'batchnorm'

  # Number of epochs between saving checkpoints. Use -1 for no checkpoints.
  config.checkpoint_interval = 25
  config.dataset = 'imagenet2012'
  # WBatch size per TPU core/GPU. The number of new datapoints gathered per
  # batch is this number divided by ensemble_size (we tile the batch by that #
  # of times).
  config.per_core_batch_size = 64
  config.num_cores = 1
  config.seed = 8
  config.train_epochs = 90
  config.log_freq = 200

  return config
