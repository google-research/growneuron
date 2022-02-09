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

r"""MobileNet-v1 on ImageNet trained with maximum likelihood.

"""

import itertools
import os
import time

from absl import app
from absl import flags
from absl import logging
from growneuron import growers
from growneuron import updaters
from growneuron.imagenet import data
from growneuron.imagenet import mb_v1
from ml_collections import config_flags
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines.schedules as ub_schedules
from tensorboard.plugins.hparams import api as hp


config_flags.DEFINE_config_file(
    name='config',
    default='growneuron/imagenet/configs/'
    'baseline_big.py',
    help_string='training config file.')
# common flags
flags.DEFINE_string(
    'tpu', '',
    'TPU address. If empty MirroredStrategy is used.')
flags.DEFINE_string('data_dir', None,
                    'data_dir to be used for tfds dataset construction.'
                    'It is required when training with cloud TPUs')
flags.DEFINE_bool('download_data', False,
                  'Whether to download data locally when initializing a '
                  'dataset.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
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
  if (hasattr(config, 'grow_frequency_multiplier') and
      config.grow_frequency_multiplier != 1):
    # Scale the frequency of the growth steps
    factor = config.grow_frequency_multiplier
    config.updater.update_frequency = int(
        config.updater.update_frequency * factor)
    config.updater.n_growth_steps = int(
        config.updater.n_growth_steps / factor)
    config.updater.n_grow_fraction *= factor

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(config.seed)
  if FLAGS.tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.MirroredStrategy()
    topology = None

  ds_builder = tfds.builder(config.dataset)
  if FLAGS.download_data:
    ds_builder.download_and_prepare()
  ds_info = ds_builder.info
  batch_size = config.per_core_batch_size * config.num_cores

  # Scale arguments that depend on 128 batch size total training iterations.
  multiplier = 512. / batch_size
  if hasattr(config.updater, 'update_frequency'):
    config.updater.update_frequency = int(
        config.updater.update_frequency * multiplier)
    config.updater.start_iteration = int(
        config.updater.start_iteration * multiplier)

  train_dataset_size = ds_info.splits['train'].num_examples
  steps_per_epoch = train_dataset_size // batch_size
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', train_dataset_size)
  steps_per_eval = ds_info.splits['validation'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  train_dataset = strategy.distribute_datasets_from_function(
      data.build_input_fn(ds_builder, batch_size, topology=topology,
                          is_training=True))
  test_dataset = strategy.distribute_datasets_from_function(
      data.build_input_fn(ds_builder, batch_size, topology=topology,
                          is_training=False))
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
    old_epochs = config.train_epochs
    # Adjust the total epochs to match big-baseline training FLOPs.
    if grower:
      config.train_epochs = updaters.adjust_epochs(
          config.train_epochs,
          config.model.width_multiplier,
          config.updater.update_frequency,
          config.updater.start_iteration,
          config.updater.n_growth_steps,
          steps_per_epoch
          )
    else:
      # baseline
      config.train_epochs = config.train_epochs / config.model.width_multiplier
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
  architecture = config.get('architecture', 'mb_v1')
  with strategy.scope():
    if architecture == 'mb_v1':
      logging.info('Building VGG model')
      model = mb_v1.create_model(
          num_classes=num_classes,
          seed=config.seed,
          **config.model)
      grow_layer_tuples = model.get_grow_layer_tuples()
    else:
      raise ValueError(f'Unknown architecture: {architecture}')
    logging.info('#grow_layer_tuples: %s', len(grow_layer_tuples))
    logging.info('grow_layer_tuples[0]: %s', grow_layer_tuples[0])
    grow_metrics = {layers[0]: tf.keras.metrics.Sum()
                    for layers in grow_layer_tuples}
    # Initialize the parameters.
    def compile_model_fn():
      model(tf.keras.Input((224, 224, 3)))
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

    # Create the updater.
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
    logging.info(message)

    if (epoch % 20 == 0) or (config.train_epochs == (epoch + 1)):
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
