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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python import debug as tfdbg 

import cnnHAR
import cnnHAR_eval

train_dir = '/home/ubuntu/deepHAR/CNN_Human_Activity_Recognition/cnnHAR_check'

num=12 # number of nodes

max_steps = num*7*4*110+1

log_device_placement = False

log_frequency = num*7*4 #

batch_size = cnnHAR.batch_size

NUM_CLASSES = cnnHAR.NUM_CLASSES
def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
        signals, labels,indices = cnnHAR.distorted_inputs()
    print('<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>')
      
    # Build a Graph that computes the logits predictions from the
    # inference model.
    #training = tf.placeholder(tf.bool)
    '''
    reshape1=cnnHAR.inference_cov11(signals)
    
    local21=cnnHAR.inference_local21(reshape1)
    local22=cnnHAR.inference_local22(reshape1)
    
    local31=cnnHAR.inference_local31(local21)
    local32=cnnHAR.inference_local32(local22)
    
    local41=cnnHAR.inference_local41(local31)
    local42=cnnHAR.inference_local42(local31)
    local43=cnnHAR.inference_local43(local32)
    '''
    reshape1=cnnHAR.inference_cov11(signals)
    reshape2=cnnHAR.inference_cov12(signals)
    
    local21=cnnHAR.inference_local21(reshape1)
    local22=cnnHAR.inference_local22(reshape1)
    local23=cnnHAR.inference_local23(reshape1)
    local24=cnnHAR.inference_local24(reshape1)
    local25=cnnHAR.inference_local25(reshape2)
    
    logits1=cnnHAR.inference1(local21,'_01')
    logits2=cnnHAR.inference1(local22,'_02')
    logits3=cnnHAR.inference1(local23,'_03')
    logits4=cnnHAR.inference1(local22,'_04')
    logits5=cnnHAR.inference1(local22,'_05')
    logits6=cnnHAR.inference1(local22,'_06')
    logits7=cnnHAR.inference1(local24,'_07')
    logits8=cnnHAR.inference1(local25,'_08')
    logits9=cnnHAR.inference1(local24,'_09')
    logits10=cnnHAR.inference1(local22,'_10')
    logits11=cnnHAR.inference1(local24,'_11')
    logits12=cnnHAR.inference1(local23,'_12')
    
    loss1=cnnHAR.loss(logits1, labels,'_01')
    loss2=cnnHAR.loss(logits2, labels,'_02')
    loss3=cnnHAR.loss(logits3, labels,'_03')
    loss4=cnnHAR.loss(logits4, labels,'_04')
    loss5=cnnHAR.loss(logits5, labels,'_05')
    loss6=cnnHAR.loss(logits6, labels,'_06')
    loss7=cnnHAR.loss(logits7, labels,'_07')
    loss8=cnnHAR.loss(logits8, labels,'_08')
    loss9=cnnHAR.loss(logits9, labels,'_09')
    loss10=cnnHAR.loss(logits10, labels,'_10')
    loss11=cnnHAR.loss(logits11, labels,'_11')
    loss12=cnnHAR.loss(logits12, labels,'_12')
    
    train_op1 = cnnHAR.train(loss1, global_step,'_01')
    train_op2 = cnnHAR.train(loss2, global_step,'_02')
    train_op3 = cnnHAR.train(loss3, global_step,'_03')
    train_op4 = cnnHAR.train(loss4, global_step,'_04')
    train_op5 = cnnHAR.train(loss5, global_step,'_05')
    train_op6 = cnnHAR.train(loss6, global_step,'_06')
    train_op7 = cnnHAR.train(loss7, global_step,'_07')
    train_op8 = cnnHAR.train(loss8, global_step,'_08')
    train_op9 = cnnHAR.train(loss9, global_step,'_09')
    train_op10 = cnnHAR.train(loss10, global_step,'_10')
    train_op11 = cnnHAR.train(loss11, global_step,'_11')
    train_op12 = cnnHAR.train(loss12, global_step,'_12')
    
    
    
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #tmp = tf.concat([labels,signals],1)
        return tf.train.SessionRunArgs(loss1)  # Asks for loss value.

      def after_run(self, run_context, run_values):
#        if self._step == 1000:
#          #tf.Session().run(tf.global_variables_initializer())
#          ndar = np.array(run_values.results)
#          np.savetxt("logits.csv", ndar.reshape(128,256), delimiter=",")
        if self._step % log_frequency == 0:
          #print('~~~~~~~~~~~~~~~~after run1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time
          
          loss_value = run_values.results
          examples_per_sec = log_frequency * batch_size / duration
          sec_per_batch = float(duration / log_frequency)

          format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
    class _LoggerHook2(tf.train.SessionRunHook):
      """Logs signals."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return tf.train.SessionRunArgs(logits)  # Asks for logits.

      def after_run(self, run_context, run_values):
        if self._step == max_steps-1:#:
          print('~~~~~~~~~~~~~~~~after run2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          ndar = np.array(run_values.results)
          np.savetxt("logits"+str(self._step)+".csv", ndar.reshape(batch_size,NUM_CLASSES), delimiter=",")

    class _LoggerHook3(tf.train.SessionRunHook):
      """Logs labels."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(labels)  # Asks for labels.

      def after_run(self, run_context, run_values):
        if self._step == max_steps-1:
          ndar = np.array(run_values.results)
          np.savetxt("labels"+str(self._step)+".csv", ndar.reshape(batch_size,NUM_CLASSES), delimiter=",")

    class _LoggerHook4(tf.train.SessionRunHook):
      """Logs signals."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return tf.train.SessionRunArgs(signals)  # Asks for signals.

      def after_run(self, run_context, run_values):
        if self._step % (10*log_frequency) == 0:
        #if self._step == max_steps-1:#:
          #print('~~~~~~~~~~~~~~~~after run4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          cnnHAR_eval.main()

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=max_steps),
               #tf.train.NanTensorHook(loss),
               _LoggerHook(),
               #_LoggerHook2(),
               _LoggerHook4()],#,save_checkpoint_steps=5000
        config=tf.ConfigProto(
            log_device_placement=log_device_placement),save_checkpoint_steps=log_frequency) as mon_sess:
      i=0
      while not mon_sess.should_stop():
#        mon_sess = tfdbg.LocalCLIDebugWrapperSession(mon_sess)
        #mon_sess.run([train_op1,extra_update_ops])
        #print('~~~~~~~~~~~~~~~~%d step:'%i)
        
        index=int(i%(num*7)/7)
        if index==0:
            #print('~~~~~~~~~~~~~~~~train_op1')
            mon_sess.run([train_op1,extra_update_ops])
        elif index==1:
            #print('~~~~~~~~~~~~~~~~train_op2')
            mon_sess.run([train_op2,extra_update_ops])
        elif index==2:
            #print('~~~~~~~~~~~~~~~~train_op3')
            mon_sess.run([train_op3,extra_update_ops])
        elif index==3:
            #print('~~~~~~~~~~~~~~~~train_op4')
            mon_sess.run([train_op4,extra_update_ops])
        elif index==4:
            #print('~~~~~~~~~~~~~~~~train_op5')
            mon_sess.run([train_op5,extra_update_ops])
        elif index==5:
            #print('~~~~~~~~~~~~~~~~train_op6')
            mon_sess.run([train_op6,extra_update_ops])
        elif index==6:
            #print('~~~~~~~~~~~~~~~~train_op1')
            mon_sess.run([train_op7,extra_update_ops])
        elif index==7:
            #print('~~~~~~~~~~~~~~~~train_op2')
            mon_sess.run([train_op8,extra_update_ops])
        elif index==8:
            #print('~~~~~~~~~~~~~~~~train_op3')
            mon_sess.run([train_op9,extra_update_ops])
        elif index==9:
            #print('~~~~~~~~~~~~~~~~train_op4')
            mon_sess.run([train_op10,extra_update_ops])
        elif index==10:
            #print('~~~~~~~~~~~~~~~~train_op5')
            mon_sess.run([train_op11,extra_update_ops])
        elif index==11:
            #print('~~~~~~~~~~~~~~~~train_op6')
            mon_sess.run([train_op12,extra_update_ops])
        i=i+1
        
        #print('~~~~~~~~~~~~~~~~one session ends~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def main(argv=None):  # pylint: disable=unused-argument
#  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

