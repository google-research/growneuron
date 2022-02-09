# coding=utf-8
# Copyright 2022 GradMax Authors.
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
from growneuron.cifar.configs import baseline_small
import ml_collections


def get_config():
  """Builds and returns config."""
  config = baseline_small.get_config()

  config.architecture = 'vgg'
  config.model = ml_collections.ConfigDict()
  config.model.depth = 11
  config.model.normalization_type = 'none'
  config.model.width_multiplier = 0.25
  config.optimizer.base_learning_rate = 0.05

  return config
