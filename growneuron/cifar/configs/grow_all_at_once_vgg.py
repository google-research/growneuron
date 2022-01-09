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

"""Default config for random growing."""
from growneuron.cifar.configs import baseline_small_vgg


def get_config():
  """Builds and returns config."""
  config = baseline_small_vgg.get_config()
  config.updater_type = 'all_at_once'
  config.grow_type = 'add_random'
  config.grow_batch_size = 128
  config.grow_epsilon = 0.
  config.is_outgoing_zero = False
  config.grow_scale_method = 'fixed'
  config.model.normalization_type = 'none'
  config.updater.carry_optimizer = True

  # We are aiming 12*2500=30000 steps growth period.
  config.updater.update_frequency = 2500
  config.updater.start_iteration = 10000
  config.scale_epochs = False
  # 1 cyle is 12 growth steps.
  config.updater.n_growth_steps = 12  # 12 cycle
  # Use one of the following
  # config.updater.n_grow = 2
  config.updater.n_grow_fraction = 0.25
  config.updater.scale = 0.5

  return config
