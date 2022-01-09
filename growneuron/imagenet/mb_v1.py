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


def check_grow_layer(layer):
  return (isinstance(layer, glayers.GrowLayer) and
          isinstance(layer.layer,
                     (tf.keras.layers.Dense, tf.keras.layers.Conv2D)) and
          not isinstance(layer.layer, tf.keras.layers.DepthwiseConv2D))


def Conv2D(filters, seed=None, **kwargs):
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 1,
      'padding': 'same',
      'strides': 1,
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def DepthwiseConv2D(seed=None, **kwargs):
  """DepthwiseConv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'strides': 1,
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.DepthwiseConv2D(**default_kwargs)


class MobilenetV1(tf.keras.Model):
  """Builds a MobileNet-v1.

  Attributes:
    width_multiplier: The number of filters in the first layer
                      ("1" corresponds to 64 filters).
    num_classes: Number of output classes.
    normalization_type: NormalizationType, of the normalization used inside
      blocks.
    l2: L2 regularization coefficient.
    seed: random seed used for initialization.
  """

  def __init__(self,
               width_multiplier,
               num_classes,
               normalization_type,
               l2,
               seed = 42):
    super().__init__(name=F'MBv1-{width_multiplier}')
    l2_reg = tf.keras.regularizers.l2

    rng_seed = [seed, seed + 1]
    rng_seed, seed = tf.random.experimental.stateless_split(rng_seed)
    self.layer_list = [
        glayers.GrowLayer(
            Conv2D(32 * width_multiplier,
                   strides=2,
                   kernel_size=3,
                   seed=seed[0],
                   kernel_regularizer=l2_reg(l2))),
        glayers.GrowLayer(BatchNormalization()),
        tf.keras.layers.Activation(glayers.get_activation_fn('relu1'))
    ]

    # MobileNet consists of blocks of convs.
    # Within each block, each conv has of base_width * multiplier filters.
    blocklist = [[1], [2, 2], [4, 4], [8, 8, 8, 8, 8, 8], [16, 16]]
    base_width = int(64 * width_multiplier)
    downsample = False
    for i, block in enumerate(blocklist):
      for j, multiplier in enumerate(block):
        rng_seed, seed = tf.random.experimental.stateless_split(rng_seed)
        self.layer_list.append(glayers.GrowLayer(DepthwiseConv2D(
            seed=seed[0],
            kernel_regularizer=tf.keras.regularizers.l2(l2))))
        self.layer_list.append(tf.keras.layers.Activation(
            glayers.get_activation_fn('relu1')))
        if normalization_type == NormalizationType.batchnorm:
          self.layer_list.append(glayers.GrowLayer(BatchNormalization()))
        elif normalization_type == NormalizationType.layernorm:
          self.layer_list.append(glayers.GrowLayer(LayerNormalization()))
        elif normalization_type == NormalizationType.none:
          pass
        else:
          raise ValueError
        rng_seed, seed = tf.random.experimental.stateless_split(rng_seed)
        # We are doing strides at conv not at dw, as it's better for
        # decomposition.
        n_channels = base_width * multiplier
        if (i+1) == len(blocklist) and (j+1) == len(block):
          # We don't scale the last layer since we are not growing it.
          n_channels = 64 * multiplier
        self.layer_list.append(glayers.GrowLayer(Conv2D(
            n_channels,
            seed=seed[0],
            strides=1 if not downsample else 2,
            kernel_regularizer=tf.keras.regularizers.l2(l2))))
        downsample = False
        self.layer_list.append(tf.keras.layers.Activation(
            glayers.get_activation_fn('relu1')),)
        self.layer_list.append(glayers.GrowLayer(BatchNormalization()))

      downsample = True
    # TODO make global pooling+dense a conv-layer, so that we can grow.
    self.layer_list.append(
        tf.keras.layers.GlobalAveragePooling2D())
    rng_seed, seed = tf.random.experimental.stateless_split(rng_seed)
    self.layer_list.append(
        tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.HeNormal(
                seed=seed[0]),
            kernel_regularizer=l2_reg(l2)
            )
        )

  def call(self, x):
    for layer in self.layer_list:
      x = layer(x)
    return x

  def get_grow_layer_tuples(self):
    """Gets all groups of layers that need to grow together."""
    grow_layers = [i for i, l in enumerate(self.layer_list)
                   if check_grow_layer(l)]

    grow_layer_tuples = []
    for i, j in zip(grow_layers[:-1], grow_layers[1:]):
      # Grow tuples should be in order.
      grow_layer_tuples.append(self.layer_list[i:(j+1)])
    return grow_layer_tuples


def create_model(
    width_multiplier = 1,
    num_classes = 1000,
    l2_coef = 0.0,
    normalization_type = 'batchnorm',
    **unused_kwargs):
  """Creates model."""
  normalization_type = NormalizationType[normalization_type]
  model = MobilenetV1(
      width_multiplier=width_multiplier,
      num_classes=num_classes,
      normalization_type=normalization_type,
      l2=l2_coef)
  return model
