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

"""Data pipeline.

Forked from simclr/tf2 codebase.
"""
import functools
from typing import Optional
from absl import logging
from growneuron.imagenet import data_util

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def build_input_fn(
    builder,
    global_batch_size,
    topology,
    is_training,
    image_size = 224):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.
    image_size: Size of the output images.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)

    preprocess_fn = get_preprocess_fn(is_training, image_size)
    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      image = preprocess_fn(image)
      return image, label

    dataset = builder.as_dataset(
        split='train' if is_training else 'validation',
        shuffle_files=is_training,
        as_supervised=True)
    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if is_training:
      buffer_multiplier = 50 if image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    prefetch_buffer_size = 2 * topology.num_tpus_per_task if topology else 2
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _input_fn


def get_preprocess_fn(is_training, image_size=224):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      image_size=image_size,
      is_training=is_training,
      test_crop=test_crop)
