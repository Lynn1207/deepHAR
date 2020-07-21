# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
SIGNAL_SIZE = 128
channels = 1
num=12


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 6
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 32*num
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 128*num/6

def read_cnnHAR(filename_queue):

  class CNNHARRecord(object):
    pass
  result = CNNHARRecord()
  
  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TextLineReader()
  result.key, value = reader.read(filename_queue)
  
  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_defaults = [[1.0] for col in range(SIGNAL_SIZE+2)]
  
  record_bytes = tf.decode_csv(value, record_defaults = record_defaults)
  #print('!!!!!!!!!!!!!!!!!!! result.type', record_bytes)
  # The first bytes represent the label, which we convert from uint8->int32.
  result.signal = tf.cast(
      tf.strided_slice(record_bytes, [0], [SIGNAL_SIZE]), tf.float32)
  result.signal = tf.reshape(result.signal, [SIGNAL_SIZE, channels])
  # labels-1 cause the logits is defaulted to start with 0~NUM_CLASS-1
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [SIGNAL_SIZE], [SIGNAL_SIZE+1])-1, tf.float32)
  #print('!!!!!!!!!!!!!!!!!!! result.label before reshape', result.label)
  result.label = tf.reshape(result.label, [1, 1])
  result.index = tf.cast(
      tf.strided_slice(record_bytes, [SIGNAL_SIZE+1], [SIGNAL_SIZE+2]), tf.float32)
  #print('!!!!!!!!!!!!!!!!!!! result.label before reshape', result.label)
  result.index = tf.reshape(result.index, [1, 1])
  
  
  return result

"""
def read_cifar10_2(filename_queue):

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TextLineReader()
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_defaults = [[1.0] for col in range(SIGNAL_SIZE+1)]
  record_bytes = tf.decode_csv(value, record_defaults = record_defaults)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [1]), tf.float32)
  result.label = tf.reshape(result.label, [1, 1])

  result.signal = tf.cast(
      tf.strided_slice(record_bytes, [1], [SIGNAL_SIZE+1]), tf.float32)
  result.signal = tf.reshape(result.signal, [SIGNAL_SIZE, channels])

  return result
"""

def _generate_image_and_label_batch(signal, label, index, min_queue_examples,
                                    batch_size, shuffle):
  print('????????? signal shape BEFORE batch', signal.get_shape())
  num_preprocess_threads = 1
  if shuffle:
    signals, label_batch,indices = tf.train.shuffle_batch(
        [signal, label,index],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    signals, label_batch,indices = tf.train.batch(
        [signal, label,index],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  print('????????? signal shape AFTER batch reshape', signals.get_shape())
  return signals, label_batch,indices #tf.reshape(label_batch, [batch_size, SIGNAL_SIZE, 1])

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filename = [os.path.join(data_dir, '0124_train.csv')]
  #if not tf.io.gfile.exists(filename):
    #raise ValueError('Failed to find file: ' + filename)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filename)
  
  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_cnnHAR(filename_queue)
    signal = read_input.signal
    signal.set_shape([SIGNAL_SIZE, channels])
    read_input.label.set_shape([1, 1])
    read_input.index.set_shape([1, 1])
    #print('?????????? singals: %f'% signal[1][0])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d acc_frames before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(signal, read_input.label,read_input.index,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

def inputs(eval_data, data_dir, batch_size):

  if not eval_data:
    filenames = [os.path.join(data_dir, '0124_train.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, '0124_test.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  #if not tf.io.gfile.exists(filenames):
      #raise ValueError('Failed to find file: ' + filenames)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cnnHAR(filename_queue)
    signal = read_input.signal

    signal.set_shape([SIGNAL_SIZE, channels])
    read_input.label.set_shape([1, 1])
    read_input.index.set_shape([1, 1])
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d acc_frames before starting to test. '
                             'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(signal, read_input.label,read_input.index,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

