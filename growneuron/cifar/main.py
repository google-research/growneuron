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

r"""Wide ResNet 28-10 on CIFAR-10/100 trained with maximum likelihood.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.

"""

import itertools
import os
import time

from absl import app
from absl import flags
from absl import logging
from growneuron import growers
from growneuron import updaters
from growneuron.cifar import data
from growneuron.cifar import vgg
from growneuron.cifar import wide_resnet
import growneuron.layers as glayers
from ml_collections import config_flags
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines.schedules as ub_schedules
from tensorboard.plugins.hparams import api as hp

config_flags.DEFINE_config_file(
    name='config',
    default='growneuron/cifar/configs/'
    'baseline_big.py',
    help_string='training config file.')
# common flags

flags.DEFINE_string('data_dir', None,
                    'data_dir to be used for tfds dataset construction.'
                    'It is required when training with cloud TPUs')
flags.DEFINE_bool('download_data', False,
                  'Whether to download data locally when initializing a '
                  'dataset.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')

FLAGS = flags.FLAGS


def get_optimizer(optimizer_config, train_epochs, batch_size, steps_per_epoch):
  """Given the config and training arguments returns an optimizer."""
  # Linearly scale learning rate and the decay epochs by vanilla settings.
  base_lr = optimizer_config.base_learning_rate * batch_size / 128
  lr_decay_epochs = [int(fraction * train_epochs)
                     for fraction in optimizer_config.lr_decay_epochs]
  if optimizer_config.decay_type == 'step':
    lr_schedule = ub_schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=optimizer_config.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=optimizer_config.lr_warmup_epochs)
  elif optimizer_config.decay_type == 'cosine':
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        base_lr, train_epochs * steps_per_epoch, alpha=0.0)
  else:
    lr_schedule = base_lr / 100.
    logging.info('No decay used')
  optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                      momentum=optimizer_config.momentum,
                                      nesterov=optimizer_config.nesterov)
  return optimizer


def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg
  config = FLAGS.config

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(config.seed)

  strategy = tf.distribute.MirroredStrategy()

  ds_builder = tfds.builder(config.dataset)
  if FLAGS.download_data:
    ds_builder.download_and_prepare()
  ds_info = ds_builder.info
  batch_size = config.per_core_batch_size * config.num_cores

  # Scale arguments that depend on 128 batch size total training iterations.
  multiplier = 128. / batch_size
  if hasattr(config.updater, 'update_frequency'):
    config.updater.update_frequency = int(
        config.updater.update_frequency * multiplier)
    config.updater.start_iteration = int(
        config.updater.start_iteration * multiplier)

  train_dataset_size = ds_info.splits['train'].num_examples
  steps_per_epoch = train_dataset_size // batch_size
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', train_dataset_size)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  train_dataset = strategy.distribute_datasets_from_function(
      data.build_input_fn(ds_builder, batch_size, topology=None,
                          is_training=True,
                          cache_dataset=config.cache_dataset))
  # Grow batches might be different in size.
  grow_batch_size = getattr(config, 'grow_batch_size', batch_size)
  grow_dataset = strategy.distribute_datasets_from_function(
      data.build_input_fn(ds_builder, grow_batch_size, topology=None,
                          is_training=True,
                          cache_dataset=config.cache_dataset))
  test_dataset = strategy.distribute_datasets_from_function(
      data.build_input_fn(ds_builder, batch_size, topology=None,
                          is_training=False,
                          cache_dataset=config.cache_dataset))
  # Scale the trianing epochs to match roughly to big-baseline cost.
  arch_name = config.get('architecture', 'wide-resnet')
  # Maybe create a grower.
  grow_type = getattr(config, 'grow_type', None)
  if grow_type == 'add_random':
    grower = growers.AddRandom()
  elif grow_type == 'add_firefly':
    grower = growers.AddFirefly()
  elif grow_type == 'add_gradmax_opt':
    grower = growers.AddGradmaxOptim()
  elif grow_type == 'add_gradmax':
    grower = growers.AddGradmax()
  else:
    logging.info('No growing')
    grower = None

  if grower:
    grower.epsilon = config.grow_epsilon
    grower.scale_method = config.grow_scale_method
    grower.is_outgoing_zero = config.is_outgoing_zero

  if config.scale_epochs:
    if arch_name == 'wide-resnet':
      width_scale = config.model.block_width_multiplier
    elif arch_name == 'vgg':
      width_scale = config.model.width_multiplier
    else:
      raise ValueError(f'Unknown architecture: {arch_name}')
    old_epochs = config.train_epochs
    # Adjust the total epochs to match big-baseline training FLOPs.
    if grower:
      config.train_epochs = updaters.adjust_epochs(
          config.train_epochs,
          width_scale,
          config.updater.update_frequency,
          config.updater.start_iteration,
          config.updater.n_growth_steps,
          steps_per_epoch
          )
    else:
      # baseline
      config.train_epochs = config.train_epochs / width_scale
    logging.info('Extended training from %s to %s', old_epochs,
                 config.train_epochs)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with summary_writer.as_default():
    flat_param_dict = {}
    def flat_fn(dic):
      for k, v in dic.items():
        if isinstance(v, dict):
          flat_fn({f'{k}.{k2}': v2 for k2, v2 in v.items()})
        else:
          flat_param_dict[k] = str(v) if isinstance(v, list) else v
    flat_fn(config.to_dict())
    hp.hparams(flat_param_dict)

  grow_layer_tuples = []
  with strategy.scope():
    if arch_name == 'wide-resnet':
      logging.info('Building ResNet model')
      model = wide_resnet.create_model(
          num_classes=num_classes,
          seed=config.seed,
          **config.model)
      for block_seq in model.group_seq:
        for block_layers, _ in block_seq:
          # We need to get all layers between the two grow layers.
          glayer_indices = [i for i, l in enumerate(block_layers)
                            if isinstance(l, glayers.GrowLayer)]
          start_index, end_index = glayer_indices[0], glayer_indices[-1]
          grow_layer_tuples.append(block_layers[start_index:(end_index+1)])
    elif arch_name == 'vgg':
      logging.info('Building VGG model')
      model = vgg.create_model(
          num_classes=num_classes,
          seed=config.seed,
          **config.model)
      grow_layer_tuples = model.get_grow_layer_tuples()
    else:
      raise ValueError(f'Unknown architecture: {arch_name}')
    logging.info('grow_layer_tuples: %s', grow_layer_tuples)
    grow_metrics = {layers[0]: tf.keras.metrics.Sum()
                    for layers in grow_layer_tuples}
    # Initialize the parameters.
    def compile_model_fn():
      model(tf.keras.Input((32, 32, 3)))
    compile_model_fn()
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    optimizer = get_optimizer(config.optimizer, config.train_epochs, batch_size,
                              steps_per_epoch)
    train_metrics = {
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
    }

    eval_metrics = {
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
    }
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # TODO This probably wouldn't work if the networks is grown;
      # so we need to switch to saved models maybe.
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    def loss_fn(inputs):
      images, labels = inputs
      logits = model(images, training=True)
      one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
      loss = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(
              one_hot_labels,
              logits,
              from_logits=True))
      scaled_loss = loss / strategy.num_replicas_in_sync
      # Don't add the regularization as unnecessary for zero variables.
      return scaled_loss

    updater_type = getattr(config, 'updater_type', None)
    if updater_type == 'round_robin':
      updater = updaters.RoundRobin(grower, grow_layer_tuples, loss_fn,
                                    compile_model_fn, **config.updater)
    elif updater_type == 'all_at_once':
      updater = updaters.AllAtOnce(grower, grow_layer_tuples, loss_fn,
                                   compile_model_fn, **config.updater)
    else:
      updater = updaters.DummyUpdater(grow_layer_tuples)

  def get_update_fn(model):
    """Returns Per-Replica update function."""
    # We need to remap this as variable names change when the network is grown.
    variable_mapped_grow_metrics = {
        l.weights[0].name: metric for l, metric in grow_metrics.items()
    }

    @tf.function
    def _update_fn(inputs):
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
        nll_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                one_hot_labels,
                logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = nll_loss + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync
      grads = tape.gradient(scaled_loss, model.trainable_variables)
      # Logging some gradient norms
      for grad, var in zip(grads, model.trainable_variables):
        if var.name in variable_mapped_grow_metrics:
          sq_grad = tf.math.pow(grad, 2)
          variable_mapped_grow_metrics[var.name].update_state(sq_grad)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      train_metrics['train/loss'].update_state(loss)
      train_metrics['train/negative_log_likelihood'].update_state(nll_loss)
      train_metrics['train/accuracy'].update_state(labels, logits)

    return _update_fn

  def train_step(iterator, grow_iterator):
    """Training StepFn."""
    # This allows retracing. We need retrace as model is changing.
    update_fn = get_update_fn(model)
    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      # Maybe grow.
      is_update = updater.is_update_iteration(optimizer.iterations)
      if is_update:
        logging.info('Growing on iteration: %s', optimizer.iterations.numpy())
        with strategy.scope():
          updater.update_network(batch_data=next(grow_iterator),
                                 optimizer=optimizer)
          compile_model_fn()
        # Regenerate the function so that the model is retracted after growing.
        update_fn = get_update_fn(model)
        logging.info('Model number of weights: %s', model.count_params())
        with summary_writer.as_default():
          logging.info('Widths after growth')
          for name, n_neuron in updater.get_grow_layer_stats():
            logging.info('%s: %d', name, n_neuron)
            tf.summary.scalar(f'n_neurons/{name}', n_neuron,
                              step=optimizer.iterations)
      # Gradient Step.
      strategy.run(update_fn, args=(next(iterator),))
      # Logging
      if is_update or optimizer.iterations % config.get('log_freq', 100) == 1:
        logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                     train_metrics['train/loss'].result(),
                     train_metrics['train/accuracy'].result() * 100)
        total_results = {name: metric.result() for name, metric in
                         train_metrics.items()}
        total_results['lr'] = optimizer.learning_rate(optimizer.iterations)
        total_results['params/total'] = model.count_params()
        for layer, metric in grow_metrics.items():
          total_results[f'grad/{layer.name}'] = metric.result()
        with summary_writer.as_default():
          for name, result in total_results.items():
            tf.summary.scalar(name, result, step=optimizer.iterations)
        for metric in itertools.chain(train_metrics.values(),
                                      grow_metrics.values()):
          metric.reset_states()

  def test_step(iterator, dataset_split, num_steps):
    """Evaluation StepFn."""
    @tf.function
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      eval_metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      eval_metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  grow_iterator = iter(grow_dataset)

  start_time = time.time()
  tb_callback = None
  if FLAGS.collect_profile:
    tb_callback = tf.keras.callbacks.TensorBoard(
        profile_batch=(100, 2000),
        log_dir=os.path.join(FLAGS.output_dir, 'logs'))
    tb_callback.set_model(model)

  for epoch in range(initial_epoch, config.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    train_start_time = time.time()
    train_step(train_iterator, grow_iterator)
    train_ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size

    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * config.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   config.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    test_iterator = iter(test_dataset)
    logging.info('Starting to run eval at epoch: %s', epoch)
    test_start_time = time.time()
    test_step(test_iterator, 'test', steps_per_eval)
    test_ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
    logging.info('Done with eval on')

    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 eval_metrics['test/negative_log_likelihood'].result(),
                 eval_metrics['test/accuracy'].result() * 100)
    total_results = {name: metric.result() for name, metric
                     in eval_metrics.items()}
    total_results['train/ms_per_example'] = train_ms_per_example
    total_results['test/ms_per_example'] = test_ms_per_example
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in eval_metrics.values():
      metric.reset_states()

    if (config.checkpoint_interval > 0 and
        (epoch + 1) % config.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)


if __name__ == '__main__':
  app.run(main)
