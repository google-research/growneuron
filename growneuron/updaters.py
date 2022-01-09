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

"""Implements controllers for updating networks.
"""
import itertools
from growneuron import growers
from growneuron import layers
import tensorflow as tf


def pad_zeros_to(tensor, new_shape):
  """Pads a tensor with zeros such that final shape is new_shape.

  It expects the new_shape to be larger than the tensor.shape.
  Zeros are added to the end of each dimension.
  Args:
    tensor: 1d, 2d, 3d tensor.
    new_shape: list of dimensions where len(new_shape) == len(tensor.shape)
  Returns:
    new tensor of shape `new_shape`.
  """
  old_shape = tensor.shape

  if len(old_shape) == 1:
    # Batchnorm or bias.
    diff_shape = [new_shape[-1] - old_shape[-1]]
    concat_axis = -1
  else:
    if old_shape[-2] == new_shape[-2]:
      # Input features are same, padding at axis=-1.
      concat_axis = -1
    else:
      concat_axis = -2
    diff_shape = list(old_shape)
    diff_shape[concat_axis] = new_shape[concat_axis] - old_shape[concat_axis]
  return tf.concat([tensor, tf.zeros(diff_shape)], axis=concat_axis)


class Updater():
  """Implements common methods.

  Updaters should be created under strategy scope or strategy should be passed
  directly.
  Attr:
    network_grower: growers.LayerGrower
    grow_layer_tuples: list of lists<glayers.GrowLayer>, candidates to be
      grown together with their outgoing weights.
    loss_fn: fn, Used to calculate loss. This function should get inputs
      as input and return loss.
    compile_fn: fn, Called to compile the model.
    update_frequency: int, Number of iterations before neurons are added.
    n_grow: int, number of neurons to grow at each growth step.
    n_grow_fraction: float, must be positive. Used together with initial width
      of candidate layers to decide n_neurons to grow at each growth step for
      each candidate separately. This approach is helpful when predicting the
      final architecture from the start as number of neurons added are fixed at
      the beginning for each layer.
    start_iteration: int, to start growing
    n_growth_steps: int, number of times the network is grown.
    scale: int, passed to the grower.grow_neurons
    carry_optimizer: bool, If true the running averages are carried to the new
      optimizer after the growth. Since variables are recreated after growth
      this is necessary.
  """

  def __init__(self, network_grower, grow_layer_tuples, loss_fn=lambda x: x,
               compile_fn=lambda: None, update_frequency=1, n_grow=1,
               n_grow_fraction=None, start_iteration=None, n_growth_steps=None,
               scale=1., carry_optimizer=True):
    assert update_frequency > 0
    assert n_grow > 0
    self._update_frequency = update_frequency
    self._carry_optimizer = carry_optimizer
    self._n_grow = n_grow
    self._n_grow_fraction = n_grow_fraction
    self._scale = scale
    if start_iteration is None:
      start_iteration = update_frequency
    self._start_iteration = start_iteration
    self.loss_fn = loss_fn
    self.compile_fn = compile_fn
    self.strategy = tf.distribute.get_strategy()
    self.network_grower = self._prepare_grower(network_grower)
    self._n_growth_steps = n_growth_steps
    self._growth_counter = 0
    self._set_grow_layer_tuples(grow_layer_tuples)

  def _prepare_grower(self, grower):
    if grower:
      grower.loss_fn = self.loss_fn
      grower.compile_fn = self.compile_fn
      grower.strategy = self.strategy
    return grower

  def copy_optimizer_slots(self, optimizer, old_variables, new_variables):
    """Copy old slots and pad with zeros for new neurons."""
    for old_var, new_var in zip(old_variables, new_variables):
      for s_name in sorted(optimizer.get_slot_names()):
        old_slot_var = optimizer.get_slot(old_var, s_name)
        new_slot_var = optimizer.get_slot(new_var, s_name)
        # This is used to retrieve the part of the new slot used for the
        # old variables. This assumes new variables are appended to the end.
        new_slot_values = pad_zeros_to(old_slot_var, new_slot_var.shape)
        new_slot_var.assign(new_slot_values)

  def delete_optimizer_slots(self, optimizer, variables):
    """Deleted old variable slots from the optimizer."""
    for old_var in variables:
      key = (old_var._shared_name if old_var._in_graph_mode
             else old_var._unique_id)
      optimizer._slots.pop(key, None)

  def _set_grow_layer_tuples(self, grow_layer_tuples):
    """Sets the tuple of layers for growing."""
    if not grow_layer_tuples:
      raise ValueError("grow_layer_tuples argument can't be empty.")
    self.grow_layer_tuples = grow_layer_tuples

    def get_n_neuron(n_neuron_initial):
      if self._n_grow_fraction:
        return int(max(1, n_neuron_initial * self._n_grow_fraction))
      else:
        return self._n_grow
    # Used to calculate n_grow per layer using grow_fraction.
    # n_neurons are decided using the initial architecture.
    self._n_grow_dict = {
        tpl[0].name: get_n_neuron(tpl[0].weights[0].shape[-1])
        for tpl in grow_layer_tuples
    }

  def is_update_iteration(self, iteration):
    assert iteration >= 0
    return ((self.network_grower is not None) and
            (iteration % self._update_frequency == 0) and
            (self._start_iteration <= iteration) and
            ((self._n_growth_steps is None) or
             (self._growth_counter < self._n_growth_steps)))

  def get_variable_list(self, grow_layer_tuple):
    return list(itertools.chain.from_iterable(
        [layer.trainable_weights for layer in grow_layer_tuple]))

  def get_grow_layer_stats(self):
    all_stats = []
    for grow_layer_tuple in self.grow_layer_tuples:
      first_layer = grow_layer_tuple[0]
      n_neuron = first_layer.get_weights()[0].shape[-1]
      all_stats.append((first_layer.layer.name, n_neuron))
    return all_stats

  def update_network(self, batch_data, optimizer=None):
    raise NotImplementedError()


class DummyUpdater(Updater):
  """Implements common methods.

  Attr:
    network_grower: growers.LayerGrower
    grow_layer_tuples: list of lists<glayers.GrowLayer>, candidates to be
      grown together with their outgoing weights.
    update_frequency: int, Number of iterations before neurons are added.
  """

  def __init__(self, grow_layer_tuples):
    super().__init__(None, grow_layer_tuples, None, None)

  def update_network(self, **kwargs):
    pass

  def is_update_iteration(self, epoch):
    del epoch
    return False

  def get_grow_layer_stats(self):
    return []


class RoundRobin(Updater):
  """Updates provided candidate layers in a round robin fashion."""

  def _next_grow_layer_tuple(self, unused_batch_data):
    next_tuple_id = self._growth_counter % len(self.grow_layer_tuples)
    self._growth_counter += 1
    return self.grow_layer_tuples[next_tuple_id]

  def update_network(self, batch_data, optimizer=None):
    """Updates the network and optimizer slots."""
    grow_layer_tuple = self._next_grow_layer_tuple(batch_data)
    old_variables = self.get_variable_list(grow_layer_tuple)
    n_new = self._n_grow_dict[grow_layer_tuple[0].name]
    self.network_grower.grow_neurons(grow_layer_tuple, batch_data,
                                     n_new=n_new, scale=self._scale)
    # Run the loss function to create new variables.
    self.compile_fn()
    new_variables = self.get_variable_list(grow_layer_tuple)
    optimizer._create_slots(new_variables)
    if self._carry_optimizer and optimizer:
      self.copy_optimizer_slots(optimizer, old_variables, new_variables)
    self.delete_optimizer_slots(optimizer, old_variables)


class AllAtOnce(Updater):
  """Grows all candidate layers at once."""

  def _get_all_grow_layer_tuples(self):
    self._growth_counter += 1
    return self.grow_layer_tuples[:]

  def update_network(self, batch_data, optimizer=None):
    """Updates the network and optimizer slots."""
    grow_layer_tuples = self._get_all_grow_layer_tuples()
    for grow_layer_tuple in grow_layer_tuples:
      old_variables = self.get_variable_list(grow_layer_tuple)
      n_new = self._n_grow_dict[grow_layer_tuple[0].name]
      self.network_grower.grow_neurons(grow_layer_tuple, batch_data,
                                       n_new=n_new, scale=self._scale)
      # Run the loss function to create new variables.
      self.compile_fn()
      new_variables = self.get_variable_list(grow_layer_tuple)
      optimizer._create_slots(new_variables)
      if self._carry_optimizer and optimizer:
        self.copy_optimizer_slots(optimizer, old_variables, new_variables)
      self.delete_optimizer_slots(optimizer, old_variables)




def adjust_epochs(train_epochs, width_scale, update_frequency,
                  start_iteration, n_growth_steps, steps_per_epoch):
  """Adjust the epochs such as the total FLOPs are same as big-baseline."""
  # Here we extend training according to the FLOP saved by starting with
  # a smaller width.
  saved_fraction = (1 - width_scale)
  # Saved before growth.
  saved_steps = saved_fraction * start_iteration
  growth_duration = (update_frequency * (n_growth_steps - 1))
  # Saved during growth (2 is because of the trianble area).
  saved_steps += saved_fraction/2 * growth_duration
  new_epochs = train_epochs + int(saved_steps / steps_per_epoch)
  return new_epochs
