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

"""GrowLayer wrapper module."""
import numpy as np
import tensorflow as tf

SUPPORTED_LAYERS = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)


def get_activation_fn(actv_fn):
  """Activation choices for the layer.

  Args:
    actv_fn: str.

  Returns:
    activation fn
  """
  if actv_fn == 'relu1':
    # This has grad(f(0))=1 instead of 0 (default implementation).
    return lambda x: tf.math.maximum(x, 0)
  elif actv_fn == 'relu2':
    # This has grad(f(0))=1.
    return lambda x: tf.math.maximum(x, -1)
  else:
    return tf.keras.activations.get(actv_fn)


class GrowLayer(tf.keras.layers.Wrapper):
  """This layer wraps keras.layers in order to support growing.

  This layer allows adding callbacks to the forward pass of a layer that will be
  called with the inputs and outputs of the underlying layer before the
  activations.

  Example Usage:
    ```
    first_layer = GrowLayer(tf.keras.layers.Dense(32))
    ```

  This layer can be used for growing neurons.
    `first_layer.add_neurons_incoming(1, new_weights='zeros')`
  """

  def __init__(self, *args, activation=None, **kwargs):
    if 'name' not in kwargs:
      # args[0] is the wrapped layer
      kwargs['name'] = f'glayer_{args[0].name}'
    super().__init__(*args, **kwargs)
    self.activation = get_activation_fn(activation)
    self.reset_callbacks()

  def add_callback(self, name, fn):
    self._callbacks[name] = fn

  def remove_callback(self, name):
    del self._callbacks[name]

  def reset_callbacks(self):
    self._callbacks = {}

  def __call__(self, inputs, *args, **kwargs):
    outputs = self.layer.__call__(inputs, *args, **kwargs)
    for _, callback_fn in self._callbacks.items():
      inputs, outputs = callback_fn(inputs, outputs)
    if self.activation:
      outputs = self.activation(outputs)
    return outputs

  def add_neurons(self, n_new, new_weights='zeros', scale=1.,
                  is_outgoing=False, scale_method='mean_norm',
                  new_bias='zeros'):
    """Adds new neurons and creates a new layer.

    New weights are scaled (if not zero) to have l2-norm equal to the mean
    l2-norm of the existing weights.
    TODO Unify splitting and adding neurons.
    Args:
      n_new: number of neurons to add.
      new_weights: 'zeros', 'random' or np.ndarray.
      scale: float, scales the new_weights multiplied with the mean norm of
        the existing weights.
      is_outgoing: bool, if true adds outgoing connections from the new neurons
        coming from previous layers. In other words number of neurons in current
        layer stays constant, but they aggregate information from n_new many
        new neurons.
      scale_method: str, Type of scaling to be used when initializing new
        neurons.
        - `mean_norm` means they are normalized using the mean norm of
        existing weights.
        - `fixed` means the weights are multiplied with scale directly.
      new_bias: str, 'zeros' or 'ones'.
    """
    old_module = self.layer
    assert old_module.built
    assert new_bias in ('zeros', 'ones')
    assert isinstance(old_module, SUPPORTED_LAYERS)
    self.layer = grow_new_layer(old_module, n_new, new_weights, scale,
                                is_outgoing=is_outgoing, new_bias=new_bias,
                                scale_method=scale_method)

  def add_neurons_identity(self, n_new):
    """Adds identity neurons for various layer types.

    Args:
      n_new: number of neurons to add.
    """
    old_module = self.layer
    assert old_module.built
    if isinstance(old_module, tf.keras.layers.BatchNormalization):
      self.layer = grow_new_bn_layer(old_module, n_new)
    elif isinstance(old_module, tf.keras.layers.LayerNormalization):
      self.layer = grow_new_ln_layer(old_module, n_new)
    elif isinstance(old_module, tf.keras.layers.DepthwiseConv2D):
      self.layer = grow_new_dw_layer(old_module, n_new)
    else:
      raise ValueError(f'layer: {old_module} of {type(old_module)} is not '
                       'supported.')


def grow_new_layer(old_module, n_new, new_weights, scale, is_outgoing=False,
                   scale_method='mean_norm', new_bias='zeros'):
  """Creates new layer after adding incoming our outgoing connections.

  Args:
    old_module: Old layer to grow from. One of layers.SUPPORTED_LAYERS.
    n_new: number of neurons to add.
    new_weights: 'zeros', 'random' or np.ndarray.
    scale: float, scales the new_weights multiplied with the mean norm of
      the existing weights.
    is_outgoing: bool, True if the outgoing connections of the new neurons are
      being added to the next layer. In this case, no new neurons are generated;
      instead existing neurons receive new incoming connections.
    scale_method: str, Type of scaling to be used when initializing new
      neurons.
      - `mean_norm` means they are normalized using the mean norm of
      existing weights.
      - `fixed` means the weights are multiplied with scale directly.
    new_bias: str, zeros or ones.
  Returns:
    layer of same type as the old_module.
  """
  old_weights = old_module.get_weights()[0]
  shape_axis = -2 if is_outgoing else -1

  if scale_method == 'mean_norm':
    magnitude_new = np.mean(norm_l2(old_weights, keep_dim=shape_axis).numpy())
    magnitude_new *= scale
  elif scale_method == 'fixed':
    # We don't use the scale of existing weights for initialization.
    magnitude_new = scale
  else:
    raise ValueError(f'Not supported scale_method, {scale_method}')

  shape_new = list(old_weights.shape)
  shape_new[shape_axis] = n_new

  if isinstance(new_weights, np.ndarray):
    assert new_weights.shape == tuple(shape_new)
    # Normalize to unit norm and then scale.
    normalized_w = normalize_l2(new_weights, axis=shape_axis).numpy()
    new_neurons = normalized_w * magnitude_new
  elif new_weights == 'random':
    normalized_w = normalize_l2(np.random.uniform(size=shape_new),
                                axis=shape_axis).numpy()
    # Normalize to unit norm and then scale.
    new_neurons = normalized_w * magnitude_new
  elif new_weights == 'zeros':
    new_neurons = np.zeros(shape_new)
  else:
    raise ValueError('new_weights: %s is not valid' % new_weights)
  new_layer_weights = [np.concatenate((old_weights, new_neurons),
                                      axis=shape_axis)]

  # Assuming bias is the second weight.
  if old_module.use_bias:
    new_bias_weights = old_module.get_weights()[1]
    if not is_outgoing:
      new_neuron_bias = (np.zeros([n_new]) if (new_bias == 'zeros') else
                         np.ones([n_new]))
      new_bias_weights = np.concatenate((new_bias_weights, new_neuron_bias),
                                        axis=0)
    new_layer_weights.append(new_bias_weights)

  common_kwargs = {
      'name': old_module.name,
      'activation': old_module.activation,
      'use_bias': old_module.use_bias
  }
  for r_name in ('kernel_regularizer', 'bias_regularizer',
                 'activity_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  n_out_new = new_layer_weights[0].shape[-1]
  if isinstance(old_module, tf.keras.layers.Dense):
    new_module = tf.keras.layers.Dense(
        n_out_new,
        weights=new_layer_weights,
        **common_kwargs)
  elif isinstance(old_module, tf.keras.layers.Conv2D):
    new_module = tf.keras.layers.Conv2D(
        n_out_new,
        kernel_size=old_module.kernel_size,
        strides=old_module.strides,
        padding=old_module.padding,
        weights=new_layer_weights,
        **common_kwargs)
  else:
    raise ValueError(f'Unexpected module: {old_module}')

  return new_module


def grow_new_ln_layer(old_module, n_new):
  """Grows a new identity LayerNormalization layer."""
  new_ln_weights = []
  # One for gamma, beta
  for i in range(2):
    old_w = old_module.get_weights()[i]
    if i == 0:  # gamma
      new_w = np.ones([n_new])
    else:  # beta
      new_w = np.zeros([n_new])
    w = np.concatenate((old_w, new_w), axis=0)
    new_ln_weights.append(w)
  common_kwargs = {
      'epsilon': old_module.epsilon
  }
  for r_name in ('gamma_regularizer', 'beta_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  return tf.keras.layers.LayerNormalization(weights=new_ln_weights,
                                            **common_kwargs)


def grow_new_bn_layer(old_module, n_new):
  """Grows a new identity BatchNormalization layer."""
  new_bn_weights = []
  # One for gamma, beta, moving_mean and moving_variance
  for i in range(4):
    old_w = old_module.get_weights()[i]
    if i in (1, 2):  # beta, moving_mean
      new_w = np.zeros([n_new])
    else:  # gamma, moving variance
      new_w = np.ones([n_new])
    w = np.concatenate((old_w, new_w), axis=0)
    new_bn_weights.append(w)
  common_kwargs = {
      'epsilon': old_module.epsilon
  }
  for r_name in ('gamma_regularizer', 'beta_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      common_kwargs[r_name] = regularizer
  return tf.keras.layers.BatchNormalization(weights=new_bn_weights,
                                            **common_kwargs)


def grow_new_dw_layer(old_module, n_new):
  """Adds identity neurosn to the depthwise convolutional layers."""
  old_weights = old_module.get_weights()[0]
  shape_new = list(old_weights.shape)
  shape_new[-2] = n_new
  new_weights = np.zeros(shape_new, dtype=old_weights.dtype)
  mid_index_x = new_weights.shape[0] // 2
  mid_index_y = new_weights.shape[1] // 2
  new_weights[mid_index_x, mid_index_y, Ellipsis] = 1.
  new_layer_weights = [np.concatenate((old_weights, new_weights),
                                      axis=-2)]

  # Assuming bias is the second weight.
  if old_module.use_bias:
    new_bias = old_module.get_weights()[1]
    new_neuron_bias = np.zeros([n_new])
    new_bias = np.concatenate((new_bias, new_neuron_bias), axis=0)
    new_layer_weights.append(new_bias)

  regularizer_kwargs = {}
  for r_name in ('kernel_regularizer', 'bias_regularizer',
                 'activity_regularizer'):
    regularizer = getattr(old_module, r_name)
    if regularizer is not None:
      regularizer_kwargs[r_name] = regularizer
  new_module = tf.keras.layers.DepthwiseConv2D(
      kernel_size=old_module.kernel_size,
      name=old_module.name,
      activation=old_module.activation,
      use_bias=old_module.use_bias,
      strides=old_module.strides,
      padding=old_module.padding,
      weights=new_layer_weights,
      **regularizer_kwargs)
  return new_module


def norm_l2(tensor, keep_dim):
  norm_axes = list(range(len(tensor.shape)))
  del norm_axes[keep_dim]
  return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2), axis=norm_axes))


def normalize_l2(tensor, axis):
  assert axis in (-2, -1)
  norm = norm_l2(tensor, axis)
  scale_recipe = '...ij,i->...ij' if (axis == -2) else '...ij,j->...ij'
  return tf.einsum(scale_recipe, tensor, 1 / norm)
