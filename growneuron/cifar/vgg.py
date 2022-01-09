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

# Lint as: python3
"""VGG Network."""

import functools
from typing import Any, Dict
from growneuron.cifar import wide_resnet
import growneuron.layers as glayers
import tensorflow as tf

NormalizationType = wide_resnet.NormalizationType

BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)

LayerNormalization = functools.partial(
    tf.keras.layers.LayerNormalization,
    epsilon=1e-5)  # using epsilon and momentum defaults from Torch


def Conv2D(filters, seed=None, **kwargs):
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      "kernel_size": 3,
      "padding": "same",
      "use_bias": False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      "kernel_initializer": tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


class VGG(tf.keras.Model):
  """Builds a VGG CNN without the FC layers at the end.

  We don't add the FC layers to stay in sync with the implementation in the
  "Firefly Neural Architecture Descent" paper.

  Attributes:
    depth: Use 11 for VGG11, 16 for VGG16, etc.,
    width_multiplier: The number of filters in the first layer
                      ("1" corresponds to 64 filters).
    num_classes: Number of output classes.
    normalization_type: NormalizationType, of the normalization used inside
      blocks.
    l2: L2 regularization coefficient.
    seed: random seed used for initialization.
  """

  def __init__(self,
               depth,
               width_multiplier,
               num_classes,
               normalization_type,
               l2,
               seed = 42):
    super().__init__(name=F"VGG-{depth}-{width_multiplier}")
    l2_reg = tf.keras.regularizers.l2

    rng_seed = [seed, seed + 1]
    assert depth == 11, "Only supporting VGG11 right now"

    # VGG consists of blocks of convs separated by downsampling.
    # Within each block, each conv has of base_width * multiplier filters.
    # This dict maps VGG-xx to a list of blocks.
    architecture = {
        11: [[1], [2], [4, 4], [8, 8], [8, 8]],
        14: [[1, 1], [2, 2], [4, 4], [8, 8], [8, 8]],
        16: [[1, 1], [2, 2], [4, 4, 4], [8, 8, 8], [8, 8, 8]],
        19: [[1, 1], [2, 2], [4, 4, 4, 4], [8, 8, 8, 8], [8, 8, 8, 8]]
    }

    blocklist = architecture[depth]
    base_width = int(64 * width_multiplier)

    downsample = False
    self.layer_list = []
    for block in blocklist:
      for multiplier in block:
        rng_seed, seed = tf.random.experimental.stateless_split(rng_seed)
        self.layer_list.append(glayers.GrowLayer(Conv2D(
            base_width*multiplier, strides=1 if not downsample else 2,
            seed=seed[0],
            kernel_regularizer=tf.keras.regularizers.l2(l2))))
        downsample = False
        self.layer_list.append(tf.keras.layers.Activation(
            glayers.get_activation_fn("relu1")),)
        if normalization_type == NormalizationType.batchnorm:
          self.layer_list.append(glayers.GrowLayer(
              BatchNormalization(
                  beta_regularizer=tf.keras.regularizers.l2(l2),
                  gamma_regularizer=tf.keras.regularizers.l2(l2))))
        elif normalization_type == NormalizationType.layernorm:
          self.layer_list.append(glayers.GrowLayer(
              LayerNormalization(
                  beta_regularizer=tf.keras.regularizers.l2(l2),
                  gamma_regularizer=tf.keras.regularizers.l2(l2))))
        elif normalization_type == NormalizationType.none:
          pass
        else:
          raise ValueError
      downsample = True
    self.layer_list.append(
        glayers.GrowLayer(
            Conv2D(num_classes, strides=2, kernel_regularizer=l2_reg(l2))))
    self.layer_list.append(tf.keras.layers.Flatten())

  def call(self, x):
    for layer in self.layer_list:
      x = layer(x)
    return x

  def get_grow_layer_tuples(self):
    """Gets all groups of layers that need to grow together."""

    grow_layers = [
        i for i, l in enumerate(self.layer_list)
        if (isinstance(l, glayers.GrowLayer) and
            isinstance(l.layer, tf.keras.layers.Conv2D))
    ]

    grow_layer_tuples = []
    for i, j in zip(grow_layers[:-1], grow_layers[1:]):
      # Grow tuples should be in order.
      grow_layer_tuples.append(self.layer_list[i:(j+1)])
    return grow_layer_tuples


def create_model(
    depth = 1,
    width_multiplier = 1,
    num_classes = 10,
    l2_coef = 0.0,
    normalization_type = "batchnorm",
    **unused_kwargs):
  """Creates model."""
  normalization_type = NormalizationType[normalization_type]
  model = VGG(
      depth=depth,
      width_multiplier=width_multiplier,
      num_classes=num_classes,
      normalization_type=normalization_type,
      l2=l2_coef)
  return model
