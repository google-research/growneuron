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

"""Data pipeline.

Forked from simclr/tf2 codebase.
"""
from typing import Optional
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def build_input_fn(
    builder,
    global_batch_size,
    topology,
    is_training,
    cache_dataset = True):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.
    cache_dataset: bool, whether to cache the dataset.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if is_training:
        image_shape = tf.shape(image)
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(
            image, image_shape[0] + 4, image_shape[1] + 4)
        image = tf.image.random_crop(image, (image_shape[0], image_shape[0], 3))
        image = tf.image.random_flip_left_right(image)
      image = tf.image.convert_image_dtype(image, tf.float32)
      return image, label

    dataset = builder.as_dataset(
        split='train' if is_training else 'test',
        shuffle_files=is_training,
        as_supervised=True)
    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if cache_dataset:
      dataset = dataset.cache()
    if is_training:
      dataset = dataset.shuffle(50000)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    prefetch_buffer_size = 2 * topology.num_tpus_per_task if topology else 2
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

  return _input_fn


