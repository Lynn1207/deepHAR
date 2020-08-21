from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import random

from six.moves import urllib
import tensorflow.compat.v1 as tf
import numpy as np

import cnnHAR_input

# Basic model parameters.
batch_size = 32
                          
data_dir = '/home/ubuntu/deepHAR/CNN_Human_Activity_Recognition/data/'
                    

# Global constants describing the CIFAR-10 data set.
SIGNAL_SIZE = cnnHAR_input.SIGNAL_SIZE
NUM_CLASSES = cnnHAR_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cnnHAR_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cnnHAR_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.05      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  global data_dir
  data_dir = data_dir
  signals, labels,indices = cnnHAR_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
  signals = tf.cast(signals, tf.float64)
  labels = tf.cast(labels, tf.float64)
  indices = tf.cast(indices, tf.float64)
  return signals, labels,indices
  
def inputs(eval_data):
    global data_dir
    data_dir = data_dir
    signals, labels,indices  = cnnHAR_input.inputs(eval_data ,data_dir=data_dir,batch_size=32)
    signals = tf.cast(signals, tf.float64)
    labels = tf.cast(labels, tf.float64)
    indices = tf.cast(indices, tf.float64)
    return signals, labels,indices

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float64
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  #i=var.op.name.find('/')-2
  if wd is not None:
    print('var_name %%%%%%%%%%%%%%%%%%%%%%%% '+var.op.name+' '+index)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
  
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float64
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
  
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss,index):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9999, name='avg')
  losses = tf.get_collection('losses'+index)
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  #Debug!!!!!!!!!!!!!!!!
 # for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    #tf.summary.scalar(l.op.name + ' (raw)', l)
    #tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
  

    
def inference_cov11(signals):
    with tf.variable_scope('conv1_01_02_03_04_05_06') as scope:
           kernel = _variable_with_weight_decay('weights',
                                                shape=[ 20, 1, 64],
                                                #shape=[3, 1, 128],
                                                stddev=5e-2,
                                                wd=None)
           biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv1d(signals, kernel, [1,1,1], padding='SAME', data_format='NWC')
           pre_activation = tf.nn.bias_add(conv, biases)
           conv1 = tf.nn.relu(pre_activation, name=scope.name)
           _activation_summary(conv1)
           print ('<<<<<<<<<<<<<<<<<<<<Shape of conv1 :',conv1.get_shape())
           
         # pool1
    pool1 = tf.nn.max_pool1d(conv1, ksize=[1,3,1], strides=[1,3,1],padding='VALID',name='pool1_01_02_03_04_05_06')
     
    reshape = tf.keras.layers.Flatten()(pool1)
    reshape = tf.cast(reshape, tf.float64)
    
    return reshape
    
def inference_cov12(signals):
    with tf.variable_scope('conv1_02_03_04_05_06') as scope:
           kernel = _variable_with_weight_decay('weights',
                                                shape=[ 20, 1, 64],
                                                #shape=[3, 1, 128],
                                                stddev=5e-2,
                                                wd=None)
           biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv1d(signals, kernel, [1,1,1], padding='SAME', data_format='NWC')
           pre_activation = tf.nn.bias_add(conv, biases)
           conv1 = tf.nn.relu(pre_activation, name=scope.name)
           _activation_summary(conv1)
           print ('<<<<<<<<<<<<<<<<<<<<Shape of conv1 :',conv1.get_shape())
           
         # pool1
    pool1 = tf.nn.max_pool1d(conv1, ksize=[1,3,1], strides=[1,3,1],padding='VALID',name='pool1_02_03_04_05_06')
     
    reshape = tf.keras.layers.Flatten()(pool1)
    reshape = tf.cast(reshape, tf.float64)
    
    return reshape
    
def inference_cov13(signals):
    with tf.variable_scope('conv1_03_04_05_06') as scope:
           kernel = _variable_with_weight_decay('weights',
                                                shape=[ 20, 1, 64],
                                                #shape=[3, 1, 128],
                                                stddev=5e-2,
                                                wd=None)
           biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv1d(signals, kernel, [1,1,1], padding='SAME', data_format='NWC')
           pre_activation = tf.nn.bias_add(conv, biases)
           conv1 = tf.nn.relu(pre_activation, name=scope.name)
           _activation_summary(conv1)
           print ('<<<<<<<<<<<<<<<<<<<<Shape of conv1 :',conv1.get_shape())
           
         # pool1
    pool1 = tf.nn.max_pool1d(conv1, ksize=[1,3,1], strides=[1,3,1],padding='VALID',name='pool1_03_04_05_06')
     
    reshape = tf.keras.layers.Flatten()(pool1)
    reshape = tf.cast(reshape, tf.float64)
    
    return reshape

def inference_local21(reshape):
    dim = reshape.get_shape()[1]
     
    with tf.variable_scope('local2_01_02_03_04_05_06') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        _activation_summary(local2)
        
    return local2
    
def inference_local22(reshape):
    dim = reshape.get_shape()[1]
     
    with tf.variable_scope('local2_02_03_04_06') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        _activation_summary(local2)
        
    return local2
    
def inference_local23(reshape):
    dim = reshape.get_shape()[1]
     
    with tf.variable_scope('local2_05') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        _activation_summary(local2)
        
    return local2
    
def inference_local24(reshape):
    dim = reshape.get_shape()[1]
     
    with tf.variable_scope('local2_5') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        _activation_summary(local2)
        
    return local2
    
def inference_local31(local2):
    with tf.variable_scope('local3_01_02_03_04_05_06') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=None)#0.004,index)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.10))
        
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        _activation_summary(local3)
        
    return local3
    
def inference_local32(local2):
    with tf.variable_scope('local3_02_03_04_06') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=None)#0.004,index)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.10))
        
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        _activation_summary(local3)
        
    return local3
    
def inference_local33(local2):
    with tf.variable_scope('local3_05') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=None)#0.004,index)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.10))
        
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        _activation_summary(local3)
        
    return local3
    
def inference_local34(local2):
    with tf.variable_scope('local3_06') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=None)#0.004,index)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.10))
        
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        _activation_summary(local3)
        
    return local3
    
def inference_local41(local3):
    with tf.variable_scope('local4_01_02_03_04_05_06') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
 
    return local4
    
def inference_local42(local3):
    with tf.variable_scope('local4_02') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
      
    return local4
    
def inference_local43(local3):
    with tf.variable_scope('local4_05') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
    
    return local4
    
def inference_local44(local3):
    with tf.variable_scope('local4_06') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
    
    return local4
    
def inference_local45(local3):
    with tf.variable_scope('local4_6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
    
    return local4
    
def inference_output1(local4):
    with tf.variable_scope('softmax_linear_01_02_03_04_05_06') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    
def inference_output2(local4):
    with tf.variable_scope('softmax_linear_02') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear

def inference_output3(local4):
    with tf.variable_scope('softmax_linear_06') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    
def inference_output4(local4):
    with tf.variable_scope('softmax_linear_06') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    
def inference_output5(local4):
    with tf.variable_scope('softmax_linear_5') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    
def inference_output6(local4):
    with tf.variable_scope('softmax_linear_6') as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    
def inference1(local4,index):
    '''
    with tf.variable_scope('conv1'+index) as scope:
           kernel = _variable_with_weight_decay('weights',
                                                shape=[ 20, 1, 64],
                                                #shape=[3, 1, 128],
                                                stddev=5e-2,
                                                wd=None)
           biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv1d(signals, kernel, [1,1,1], padding='SAME', data_format='NWC')
           pre_activation = tf.nn.bias_add(conv, biases)
           conv1 = tf.nn.relu(pre_activation, name=scope.name)
           _activation_summary(conv1)
           print ('<<<<<<<<<<<<<<<<<<<<Shape of conv1 :',conv1.get_shape())
           
         # pool1
    pool1 = tf.nn.max_pool1d(conv1, ksize=[1,3,1], strides=[1,3,1],padding='VALID',name='pool1'+index)
    
    reshape = tf.keras.layers.Flatten()(pool1)
    reshape = tf.cast(reshape, tf.float64)
    
    dim = reshape.get_shape()[1]
     
    with tf.variable_scope('local2'+index) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        _activation_summary(local2)
    
    with tf.variable_scope('local3'+index) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=None)#0.004,index)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.10))
        
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        _activation_summary(local3)
    
    with tf.variable_scope('local4'+index) as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 30], stddev=0.04, wd=None)
        biases = _variable_on_cpu('biases', [30], tf.constant_initializer(0.10))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
    '''
    with tf.variable_scope('softmax_linear'+index) as scope:
          weights = _variable_with_weight_decay('weights', [30, NUM_CLASSES],stddev=1/30.0, wd=None)
          biases = _variable_on_cpu('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.matmul(local4, weights)+biases,name=scope.name)
          _activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear

def loss(logits, labels,index):
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [batch_size,1])
    logits = tf.reshape(logits, [batch_size,1,NUM_CLASSES])
    #print('loss@@@@@@@@@@@@##############',logits.get_shape())
    #print('loss@@@@@@@@@@@@##############',labels.get_shape())
    i=0
    loss=0.0
    while i<batch_size:
        loss+=-tf.math.log(logits[i,0,labels[i,0]])
        i+=1
    loss=loss/batch_size
    
    #print('loss@@@@@@@@@@@@##############',loss)
    tf.add_to_collection('losses'+index, loss)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss=tf.add_n(tf.get_collection('losses'+index),name='total_loss')
    return total_loss
    
    
def train(total_loss, global_step,index):#index is a string e.g. '_1'
 
 num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
 decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
 print('<<<<<<<<<<<<<<<<<<<<train: total_loss'+index)
 # Decay the learning rate exponentially based on the number of steps.
 lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                decay_steps,#10000,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
 tf.summary.scalar('learning_rate', lr)
 
 var_list=[]
 for var in tf.trainable_variables():
     if var.op.name.find(index)!= -1:
        var_list.append(var)
        print('@@@@@@@@@@@@@@@@@@'+var.op.name)
        '''
        if var.op.name.find('weights')!= -1 and var.op.name.find('conv')==-1 and var.op.name.find('soft')==-1:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            total_loss+= weight_decay
        '''
        
 # Generate moving averages of all losses and associated summaries.
 loss_averages_op = _add_loss_summaries(total_loss,index)

 # Compute gradients.
 with tf.control_dependencies([loss_averages_op]):
  opt = tf.train.MomentumOptimizer(lr, 0.5)#opt = tf.train.AdadeltaOptimizer(lr)
  print('#########################',opt)
  grads = opt.compute_gradients(total_loss,var_list)
  for i in range(0,len(grads)):
    print(i)
    print('<<<<<<<<<<<<<<<<< shape of grads:',grads[i])

# Apply gradients.
 apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

# Add histograms for trainable variables.
 for var in tf.trainable_variables():
  tf.summary.histogram(var.op.name, var)
  print(var.op.name)

# Add histograms for gradients.
 for grad, var in grads:
  if grad is not None:
    tf.summary.histogram(var.op.name + '/gradients', grad)

# Track the moving averages of all trainable variables.
 variable_averages = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step)
 with tf.control_dependencies([apply_gradient_op]):
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
 return variables_averages_op
