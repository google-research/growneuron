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

"""Tests for growneuron.updaters."""

import itertools
import absl.testing.parameterized as parameterized
from growneuron import growers
from growneuron import updaters
import tensorflow as tf


class RoundRobinScheduleTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.all_layers = []
    for _ in range(5):
      layer = tf.keras.layers.Dense(2)
      layer.build((None, 8))
      self.all_layers.append(layer)

  def get_random_grower(self):
    return growers.AddRandom()

  def get_grow_layers(self, n=1):
    return list(itertools.permutations(self.all_layers, 2))[:n]

  @parameterized.named_parameters(
      ('every_100_start0_3times', 100, 0, 3,
       [(0, True), (4, False), (100, True), (2300, True), (100, False)]),
      ('every_1_start0_21times', 1, 0, 21,
       [(0, True), (100, True), (2300, True), (3, True)]),
      ('every_1_start2_5times', 1, 2, 4,
       [(1, False), (2, True), (100, True), (1, False), (3, True)]),
      ('every_100_start50_1times', 100, 50, 1,
       [(20, False), (75, False), (200, True), (234, False), (300, False)]),
      ('none_start', 5, None, 25,
       [(2, False), (5, True), (7, False), (45, True)]),
  )
  def test_update_iter(self, update_frequency, start_iteration, n_growth_steps,
                       iterations):
    network_grower = self.get_random_grower()
    grow_layer_tuples = self.get_grow_layers()

    updater = updaters.RoundRobin(
        network_grower,
        grow_layer_tuples,
        update_frequency=update_frequency,
        n_grow=1,
        start_iteration=start_iteration,
        n_growth_steps=n_growth_steps)
    for iteration, bool_val in iterations:
      print(f'COUNT:{updater._growth_counter}, {iteration}, {bool_val}')
      self.assertEqual(updater.is_update_iteration(iteration), bool_val)
      if bool_val:
        updater._next_grow_layer_tuple(None)

  @parameterized.named_parameters(('n4', 4), ('n1', 1))
  def test_next_grow_layers(self, n_tuples):
    network_grower = self.get_random_grower()
    grow_layer_tuples = self.get_grow_layers(n=n_tuples)
    updater = updaters.RoundRobin(
        network_grower, grow_layer_tuples, update_frequency=2)
    for tup in itertools.chain(grow_layer_tuples, grow_layer_tuples):
      self.assertEqual(tup, updater._next_grow_layer_tuple(None))

  @parameterized.named_parameters(('1d', (2,), (3,)),
                                  ('2d_out', (2, 4), (2, 5)),
                                  ('2d_in', (2, 4), (4, 4)),
                                  # Deptwise kernel
                                  ('3d_in', (4, 4, 5), (4, 4, 8)),
                                  ('4d_in', (4, 4, 2, 3), (4, 4, 4, 3)),
                                  ('4d_out', (4, 4, 2, 3), (4, 4, 2, 5))
                                  )
  def test_pad_zeros_to(self, old_shape, new_shape):
    tensor = tf.random.uniform(old_shape)
    new_tensor = updaters.pad_zeros_to(tensor, new_shape)
    old_slice = tuple(slice(None, x) for x in tensor.shape)
    self.assertAllEqual(new_tensor[old_slice], tensor)

  @parameterized.named_parameters(
      ('dense_outgrown', (3, 4), (3, 5), lambda a, i: a[:, i]),
      ('dense_ingrown', (3, 4), (4, 4), lambda a, i: a[i, :]),
      ('conv2d_ingrown', (2, 2, 3, 4), (2, 2, 4, 4),
       lambda a, i: a[:, :, i, :]),
      ('conv2d_outgrown', (2, 2, 3, 4), (2, 2, 3, 5), lambda a, i: a[Ellipsis, i]),
      ('conv_dw', (2, 2, 3), (2, 2, 4), lambda a, i: a[:, :, i])
      )
  def test_copy_adam_slots(self, old_shape, new_shape, slice_fn):
    grow_layer_tuples = self.get_grow_layers()
    network_grower = self.get_random_grower()
    updater = updaters.RoundRobin(
        network_grower, grow_layer_tuples, update_frequency=2)
    old_var = tf.Variable(tf.ones(old_shape))
    new_var = tf.Variable(tf.ones(new_shape))
    optimizer = tf.keras.optimizers.Adam()
    optimizer._create_slots([old_var, new_var])
    random_slot_vals = tf.random.uniform(old_shape)
    for s_name in optimizer.get_slot_names():
      self.assertAllEqual(optimizer.get_slot(new_var, s_name),
                          tf.zeros(new_shape))
      optimizer.get_slot(old_var, s_name).assign(random_slot_vals)

    updater.copy_optimizer_slots(optimizer, [old_var], [new_var])
    for s_name in optimizer.get_slot_names():
      # Check new_values still have zeros.
      new_values_slice = slice_fn(optimizer.get_slot(new_var, s_name), -1)
      self.assertAllEqual(new_values_slice, tf.zeros_like(new_values_slice))
      # Check old variables have their random values set correctly.
      old_values_slice = slice_fn(optimizer.get_slot(old_var, s_name), 0)
      new_values_slice = slice_fn(optimizer.get_slot(new_var, s_name), 0)
      self.assertAllEqual(new_values_slice, old_values_slice)


if __name__ == '__main__':
  tf.test.main()
