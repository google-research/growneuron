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
"""Wide Residual Network."""

import enum
import functools
from typing import Any, Dict
import growneuron.layers as glayers
import tensorflow as tf

BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)

LayerNormalization = functools.partial(
    tf.keras.layers.LayerNormalization,
    epsilon=1e-5)  # using epsilon and momentum defaults from Torch


@enum.unique
class NormalizationType(enum.Enum):
  """Direction along the z-axis."""
  layernorm = 'layernorm'
  batchnorm = ' batchnorm'
  none = 'none'


def Conv2D(filters, seed=None, **kwargs):
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(
    filters,
    block_width,
    normalization_type,
    strides,
    l2,
    seed):
  """Basic residual block of two 3x3 convs.

  Args:
    filters: Number of filters for Conv2D.
    block_width: Multiplies the first filter.
    normalization_type: NormalizationType
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    seed: random seed used for initialization.

  Returns:
    block_layers: list of sequential layers for the main branch.
    skip_layer: tf.keras.Conv2D or None.
  """
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]

  block_layers = [
      BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2)),
      tf.keras.layers.Activation('relu'),
      glayers.GrowLayer(
          Conv2D(int(filters*block_width), strides=strides, seed=seeds[0],
                 kernel_regularizer=tf.keras.regularizers.l2(l2)))
  ]
  # Maybe add normalization in between the layers.
  if normalization_type == NormalizationType.batchnorm:
    block_layers.append(glayers.GrowLayer(
        BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.keras.regularizers.l2(l2))))
  elif normalization_type == NormalizationType.layernorm:
    block_layers.append(glayers.GrowLayer(
        LayerNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.keras.regularizers.l2(l2))))
  elif normalization_type == NormalizationType.none:
    pass
  else:
    raise ValueError

  block_layers += [
      # This is to ensure gradient is 1 at 0 for relu.
      tf.keras.layers.Activation(glayers.get_activation_fn('relu1')),
      glayers.GrowLayer(
          Conv2D(filters, strides=1, seed=seeds[1],
                 kernel_regularizer=tf.keras.regularizers.l2(l2)))
  ]

  if strides > 1:
    skip_layer = Conv2D(filters, kernel_size=1, strides=strides, seed=seeds[2],
                        kernel_regularizer=tf.keras.regularizers.l2(l2))
  else:
    skip_layer = None
  return (block_layers, skip_layer)


class WideResnet(tf.keras.Model):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Attributes:
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    block_width_multiplier: Multiplies the filters in the first conv for each
      block.
    normalization_type: NormalizationType, of the normalization used inside
      blocks.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    seed: random seed used for initialization.

  """

  def __init__(
      self,
      depth,
      width_multiplier,
      block_width_multiplier,
      normalization_type,
      num_classes,
      l2,
      seed = 42
      ):
    super().__init__(name='wide_resnet-{}-{}'.format(depth, width_multiplier))
    l2_reg = tf.keras.regularizers.l2

    seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
    if (depth - 4) % 6 != 0:
      raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
    num_blocks = (depth - 4) // 6

    self.conv_stem = Conv2D(16,
                            strides=1,
                            seed=seeds[0],
                            kernel_regularizer=l2_reg(l2))
    self.group_seq = []
    for i, (filters, strides, seed) in enumerate(
        zip([16, 32, 64], [1, 2, 2], seeds[1:4])):
      block_seq = []
      group_seeds = tf.random.experimental.stateless_split(
          [seed, seed + 1], num_blocks)[:, 0]
      for j, group_seed in enumerate(group_seeds):
        block_strides = strides if j == 0 else 1
        block_seq.append(
            basic_block(filters=filters*width_multiplier,
                        block_width=block_width_multiplier,
                        normalization_type=normalization_type,
                        strides=block_strides, l2=l2, seed=group_seed)
            )
      self.group_seq.append(block_seq)

    self.final_layers = [
        BatchNormalization(beta_regularizer=l2_reg(l2),
                           gamma_regularizer=l2_reg(l2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.AveragePooling2D(pool_size=8),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
            kernel_regularizer=l2_reg(l2),
            bias_regularizer=l2_reg(l2))
    ]

  def call(self, inputs):
    x = self.conv_stem(inputs)
    for block_seq in self.group_seq:
      for block_layers, skip_layer in block_seq:
        y = x
        # Main branch.
        for layer in block_layers:
          y = layer(y)
        # Skip branch
        if skip_layer:
          x = skip_layer(x)
        x = x + y
    for layer in self.final_layers:
      x = layer(x)
    return x


def create_model(
    depth = 22,
    width_multiplier = 1,
    block_width_multiplier = 1.,
    normalization_type = 'batchnorm',
    num_classes = 10,
    l2_coef = 0.0,
    **unused_kwargs):
  """Creates model."""
  normalization_type = NormalizationType[normalization_type]
  model = WideResnet(depth=depth,
                     width_multiplier=width_multiplier,
                     block_width_multiplier=block_width_multiplier,
                     num_classes=num_classes,
                     normalization_type=normalization_type,
                     l2=l2_coef)
  return model
