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

"""Tests for growneuron.layers."""
import absl.testing.parameterized as parameterized
import growneuron.layers as glayers
import tensorflow as tf


class LayerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('dense', tf.keras.layers.Dense(3), (3, 4)),
      ('batchnorm', tf.keras.layers.BatchNormalization(), (2, 4)),
      ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4))
  )
  def test_consistency(self, layer, input_shape):
    wrapped_layer = glayers.GrowLayer(layer)
    x = tf.random.uniform(input_shape)
    original_out = layer(x)
    new_out = wrapped_layer(x)
    self.assertAllEqual(original_out, new_out)

  @parameterized.named_parameters(
      ('dense', tf.keras.layers.Dense(3), (3, 4), 1),
      ('dense_5neuron', tf.keras.layers.Dense(3), (3, 4), 5),
      ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 1),
      ('conv2d_5neuron', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 5),
  )
  def test_add_neurons_incoming_zeros(self, layer, input_shape, n_new):
    wrapped_layer = glayers.GrowLayer(layer)
    x = tf.random.uniform(input_shape)
    original_out = wrapped_layer(x)
    old_output_shape = original_out.get_shape()
    n_neurons_old = old_output_shape[-1]
    wrapped_layer.add_neurons(n_new, new_weights='zeros', is_outgoing=False)
    new_out = wrapped_layer(x)
    # Check the output has the expected shape
    new_shape = old_output_shape[:-1] + [n_neurons_old+n_new]
    self.assertAllEqual(new_shape, new_out.get_shape())
    # Check the old neurons create same output
    self.assertAllClose(original_out, new_out[Ellipsis, :n_neurons_old])
    # Check the new neurons create zero output
    self.assertEqual(0, tf.math.count_nonzero(new_out[Ellipsis, n_neurons_old:]))
    new_weights, new_biases = wrapped_layer.get_weights()
    # Check the new weights are zero
    added_weights = new_weights[Ellipsis, n_neurons_old:]
    self.assertAllEqual(added_weights, tf.zeros_like(added_weights))
    # Check the new biases are zero
    added_biases = new_biases[n_neurons_old:]
    self.assertAllEqual(added_biases, tf.zeros_like(added_biases))

  @parameterized.named_parameters(
      ('dense', tf.keras.layers.Dense(3), (3, 4), 1),
      ('dense_5neuron', tf.keras.layers.Dense(3), (3, 4), 5),
      ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 1),
      ('conv2d_5neuron', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 5),
  )
  def test_add_neurons_outgoing_zeros(self, layer, input_shape, n_new):
    wrapped_layer = glayers.GrowLayer(layer)
    n_features = input_shape[-1]
    x = tf.random.uniform(input_shape)
    # New input after growing would have more features
    new_input_shape = input_shape[:-1] + (n_new,)
    new_x = tf.concat([x, tf.random.uniform(new_input_shape)], axis=-1)
    original_out = layer(x)
    old_weights, old_biases = wrapped_layer.get_weights()
    wrapped_layer.add_neurons(n_new, new_weights='zeros', is_outgoing=True)
    new_out = wrapped_layer(new_x)
    new_weights, new_biases = wrapped_layer.get_weights()
    print(new_weights, new_biases)
    # Output of the layer shouldn't change.
    self.assertAllClose(original_out, new_out)
    # Check biases are unchanged
    self.assertAllEqual(old_biases, new_biases)
    # Check the new weights are zero
    added_weights = new_weights[Ellipsis, n_features:, :]
    self.assertAllEqual(added_weights, tf.zeros_like(added_weights))
    # Check the old weights are same
    kept_weights = new_weights[Ellipsis, :n_features, :]
    self.assertAllEqual(old_weights, kept_weights)

  @parameterized.named_parameters(
      ('dense_kernel', 'dense', ('kernel',)),
      ('dense_bias', 'dense', ('bias',)),
      ('dense_activity', 'dense', ('activity',)),
      ('dense_all', 'dense', ('kernel', 'bias', 'activity')),
      ('conv2d_kernel', 'conv2d', ('kernel',)),
      ('conv2d_bias', 'conv2d', ('bias',)),
      ('conv2d_activity', 'conv2d', ('activity',)),
      ('conv2d_all', 'conv2d', ('kernel', 'bias', 'activity')),
  )
  def test_regularizer_incoming(self, layer_type, regularizer_types):
    reg_kwargs = {f'{r_type}_regularizer': tf.keras.regularizers.L2(0.1)
                  for r_type in regularizer_types}
    print(reg_kwargs)
    if layer_type == 'dense':
      layer = tf.keras.layers.Dense(3, **reg_kwargs)
      input_shape = (3, 4)
    elif layer_type == 'conv2d':
      layer = tf.keras.layers.Conv2D(3, 3, **reg_kwargs)
      input_shape = (3, 5, 5, 4)
    else:
      raise ValueError('not supported')
    wrapped_layer = glayers.GrowLayer(layer)
    x = tf.random.uniform(input_shape)
    _ = wrapped_layer(x)
    old_losses = wrapped_layer.losses
    wrapped_layer.add_neurons(1, new_weights='zeros', is_outgoing=False)
    _ = wrapped_layer(x)
    new_losses = wrapped_layer.losses
    for old_loss, new_loss in zip(old_losses, new_losses):
      self.assertAllClose(old_loss, new_loss)

  @parameterized.named_parameters(
      ('dense_kernel', 'dense', ('kernel',)),
      ('dense_bias', 'dense', ('bias',)),
      ('dense_activity', 'dense', ('activity',)),
      ('dense_all', 'dense', ('kernel', 'bias', 'activity')),
      ('conv2d_kernel', 'conv2d', ('kernel',)),
      ('conv2d_bias', 'conv2d', ('bias',)),
      ('conv2d_activity', 'conv2d', ('activity',)),
      ('conv2d_all', 'conv2d', ('kernel', 'bias', 'activity')),
      ('bn_beta', 'bn', ('beta',)),
  )
  def test_regularizer_outgoing(self, layer_type, regularizer_types):
    reg_kwargs = {f'{r_type}_regularizer': tf.keras.regularizers.L2(0.1)
                  for r_type in regularizer_types}
    print(reg_kwargs)
    if layer_type == 'dense':
      layer = tf.keras.layers.Dense(3, **reg_kwargs)
      input_shape = (3, 4)
    elif layer_type == 'conv2d':
      layer = tf.keras.layers.Conv2D(3, 3, **reg_kwargs)
      input_shape = (3, 5, 5, 4)
    elif layer_type == 'bn':
      layer = tf.keras.layers.BatchNormalization(**reg_kwargs)
      input_shape = (3, 4)
    else:
      raise ValueError('not supported')
    wrapped_layer = glayers.GrowLayer(layer)
    x = tf.random.uniform(input_shape)
    _ = wrapped_layer(x)
    old_losses = wrapped_layer.losses
    if layer_type == 'bn':
      wrapped_layer.add_neurons_identity(1)
    else:
      wrapped_layer.add_neurons(1, new_weights='zeros', is_outgoing=True)
    new_input_shape = input_shape[:-1] + (1,)
    new_x = tf.concat([x, tf.random.uniform(new_input_shape)], axis=-1)
    _ = wrapped_layer(new_x)
    new_losses = wrapped_layer.losses
    for old_loss, new_loss in zip(old_losses, new_losses):
      self.assertAllClose(old_loss, new_loss)

  @parameterized.named_parameters(
      ('2d_axis1', (4, 5), -1),
      ('3d_axis1', (3, 3, 1), -1),
      ('4d_axis1', (3, 3, 4, 5), -1),
      ('2d_axis2', (4, 5), -2),
      ('3d_axis2', (3, 3, 1), -2),
      ('4d_axis2', (3, 3, 4, 5), -2),
  )
  def test_norm_l2(self, shape, axis):
    tensor = tf.reshape(tf.range(tf.math.reduce_prod(shape),
                                 dtype=tf.float32), shape)
    calculated_norm = glayers.norm_l2(tensor, axis)
    if axis == -2:
      tensor = tf.einsum('...ij->...ji', tensor)
    # L2 norm should be 1 over axis 1
    flat_tensor = tf.reshape(tensor,
                             [-1, tensor.shape[-1]])
    expected_norms = tf.norm(flat_tensor, axis=-2)
    self.assertAllClose(expected_norms, calculated_norm)
    pass

  @parameterized.named_parameters(
      ('2d_axis1', (4, 5), -1),
      ('3d_axis1', (3, 3, 1), -1),
      ('4d_axis1', (3, 3, 4, 5), -1),
      ('2d_axis2', (4, 5), -2),
      ('3d_axis2', (3, 3, 1), -2),
      ('4d_axis2', (3, 3, 4, 5), -2),
  )
  def test_normalize_l2(self, shape, axis):
    tensor = tf.reshape(tf.range(tf.math.reduce_prod(shape),
                                 dtype=tf.float32), shape)
    normalized_tensor = glayers.normalize_l2(tensor, axis)
    if axis == -2:
      normalized_tensor = tf.einsum('...ij->...ji', normalized_tensor)
    # L2 norm should be 1 over axis 1
    flat_tensor = tf.reshape(normalized_tensor,
                             [-1, normalized_tensor.shape[-1]])
    norms = tf.norm(flat_tensor, axis=-2)
    self.assertAllClose(norms, tf.ones_like(norms))


if __name__ == '__main__':
  tf.test.main()
