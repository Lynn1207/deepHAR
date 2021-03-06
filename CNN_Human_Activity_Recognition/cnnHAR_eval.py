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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow.compat.v1 as tf

import cnnHAR

num=6 #number of nodes

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/deepHAR/CNN_Human_Activity_Recognition/cnnHAR_e',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/deepHAR/CNN_Human_Activity_Recognition/cnnHAR_check',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 64*num*2,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

batch_size = 32
NUM_CLASSES = cnnHAR.NUM_CLASSES

def eval_once(saver,summary_writer,labels,loss1,logits1,loss2,logits2,loss3,logits3,loss4,logits4,loss5,logits5,loss6,logits6,summary_op):
  
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('~~~~~~~~~~~checkpoint file found at step %s'% global_step)
    else:
      print('No checkpoint file found')
      return
    
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
        
      num_iter = int(math.ceil(FLAGS.num_examples/batch_size))
      cnt = 0
      step = 0
      accuracy=0
      accuracies=np.zeros(num*2)
      cnts=np.zeros(num*2)
      steps=np.zeros(num*2)
      comsample_labels=np.zeros(2*batch_size)
      simpleness=np.zeros((num,2*batch_size))
      concur_s=np.zeros((num,num))
      while step < num_iter and not coord.should_stop():
        #print('!!!!!!the step', step)
        
        #local test
        
        if int(step/2)==0:
            #print('~~~~loss1')
            samplelabels,predictions,precision=sess.run([labels,logits1,loss1])
        elif int(step/2)==1:
            #print('~~~~loss2')
            samplelabels,predictions,precision=sess.run([labels,logits2,loss2])
        elif int(step/2)==2:
            #print('~~~~loss3')
            samplelabels,predictions,precision=sess.run([labels,logits3,loss3])
        elif int(step/2)==3:
            #print('~~~~loss4')
            samplelabels,predictions,precision=sess.run([labels,logits4,loss4])
        elif int(step/2)==4:
            #print('~~~~loss5')
            samplelabels,predictions,precision=sess.run([labels,logits5,loss5])
        elif int(step/2)==5:
            #print('~~~~loss6')
            samplelabels,predictions,precision=sess.run([labels,logits6,loss6])
        
        
        #test on 7
        if int(step/2)==6:
            #print('~~~~loss1')
            samplelabels,predictions,precision=sess.run([labels,logits1,loss1])
        elif int(step/2)==7:
            #print('~~~~loss2')
            samplelabels,predictions,precision=sess.run([labels,logits2,loss2])
        elif int(step/2)==8:
            #print('~~~~loss3')
            samplelabels,predictions,precision=sess.run([labels,logits3,loss3])
        elif int(step/2)==9:
            #print('~~~~loss4')
            samplelabels,predictions,precision=sess.run([labels,logits4,loss4])
        elif int(step/2)==10:
            #print('~~~~loss5')
            samplelabels,predictions,precision=sess.run([labels,logits5,loss5])
        elif int(step/2)==11:
            #print('~~~~loss6')
            samplelabels,predictions,precision=sess.run([labels,logits6,loss6])
        
        
       
        #print('!!!!!!the index of t/????????????????/he whole batch %f /n' % output3[0][0][0])
        """
        if step == 5:         
          ndar = np.array(output)
          np.savetxt("testout.csv", ndar.reshape(128,256), delimiter=",")
          ndar = np.array(output2)
          np.savetxt("testlabel.csv", ndar.reshape(128,256), delimiter=",")
          ndar = np.array(output3)
          np.savetxt("testsignal.csv", ndar.reshape(128,256), delimiter=",")
        """
        #print(samplelabels.shape)
        #print(predictions.shape)
        n_acc=0
        l=0.0
        for i in range(0, batch_size):
            #print('label:',samplelabels.shape)
            #print('prediction:',predictions.shape)
            if int(samplelabels[i][0][0])==np.argmax(predictions[i]):
                n_acc=n_acc+1
            
            i_loss=-math.log(predictions[i][int(samplelabels[i][0][0])])
            if int(step/2)>=num:
                simpleness[int(step/2)-num][int(step%2)*batch_size+i]=i_loss
                comsample_labels[int(step%2)*batch_size+i]=samplelabels[i][0][0]
                '''
                if l<precision:
                    simpleness[int(step/2)][int(step%2)*batch_size+i]=1
                '''
            l+=i_loss
        accuracies[int(step/2)]+=100.0*n_acc/(2*batch_size)
        cnts[int(step/2)]+=l/(2*batch_size)
        steps[int(step/2)]+=1
        step += 1
      #compute the simpleness matrix
      #print('::::::::::::comsample_labels: ')
      #print(comsample_labels)
      w_loss=[0.17, 0.17, 0.14, 0.22, 0.14, 0.16]
      for i in range(0, num):
        for j in range(0,2*batch_size):
          #error magin
          simpleness[i][j]=1-simpleness[i][j]
          '''
          if simpleness[i][j]<m_loss[i]:
            simpleness[i][j]=1
          else:
            simpleness[i][j]=0
          '''
      #compute concurrent_simpleness
      for i in range(0,num):
          for j in range(0,num):
              if i!=j:
                  #concurrent simpleness between user_i and user_j
                  for n in range(0, 2*batch_size):
                      concur_s[i][j]+=((simpleness[i][n]-simpleness[j][n])*100.0)*((simpleness[i][n]-simpleness[j][n])*100.0)
                  concur_s[i][j]=math.sqrt(concur_s[i][j])/(2*batch_size)
                  #difference[i][j]=1-concur_s[i][j]
      print('concurrent simpleness: ')
      print(concur_s)
      
      i=0
      while i<num:
            print('!!!!!!!!!!!!!!!!!!!! subject %s (%s records): test loss = %.3f, accuracy=%.3f' % (i+1, steps[i],cnts[i],accuracies[i]))
            i+=1
        
      print('(locally test)!!!!!!!!!!!!!!!!!!!! average_test loss = %.3f, average_accuracy=%.3f' % (np.mean(cnts[0:num]),np.mean(accuracies[0:num])))
      
      print('(test on 7)!!!!!!!!!!!!!!!!!!!! average_test loss = %.3f, average_accuracy=%.3f' % ( np.mean(cnts[num:2*num]),np.mean(accuracies[num:2*num])))
    
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='loss @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    signals, labels = cnnHAR.inputs(eval_data=eval_data)
    
    # Build a Graph that computes the logits predictions from the

    pool11=cnnHAR.inference_cov11(signals,'_01')
    reshape1=cnnHAR.inference_cov21(pool11,'_01')
    local21=cnnHAR.inference_local21(reshape1,'_01')
    local31=cnnHAR.inference_local31(local21,'_01')
    local41=cnnHAR.inference_local41(local31,'_01')
    logits1=cnnHAR.inference_output1(local41,'_01')
    
    
    pool12=cnnHAR.inference_cov11(signals,'_02')
    reshape2=cnnHAR.inference_cov21(pool12,'_02')
    local22=cnnHAR.inference_local21(reshape2,'_02')
    local32=cnnHAR.inference_local31(local22,'_02')
    local42=cnnHAR.inference_local41(local32,'_02')
    logits2=cnnHAR.inference_output1(local42,'_02')
    
    
    pool13=cnnHAR.inference_cov11(signals,'_03')
    reshape3=cnnHAR.inference_cov21(pool13,'_03')
    local23=cnnHAR.inference_local21(reshape3,'_03')
    local33=cnnHAR.inference_local31(local23,'_03')
    local43=cnnHAR.inference_local41(local33,'_03')
    logits3=cnnHAR.inference_output1(local43,'_03')
    
    
    pool14=cnnHAR.inference_cov11(signals,'_04')
    reshape4=cnnHAR.inference_cov21(pool14,'_04')
    local24=cnnHAR.inference_local21(reshape4,'_04')
    local34=cnnHAR.inference_local31(local24,'_04')
    local44=cnnHAR.inference_local41(local34,'_04')
    logits4=cnnHAR.inference_output1(local44,'_04')

    
    pool15=cnnHAR.inference_cov11(signals,'_05')
    reshape5=cnnHAR.inference_cov21(pool15,'_05')
    local25=cnnHAR.inference_local21(reshape5,'_05')
    local35=cnnHAR.inference_local31(local25,'_05')
    local45=cnnHAR.inference_local41(local35,'_05')
    logits5=cnnHAR.inference_output1(local45,'_05')

    pool16=cnnHAR.inference_cov11(signals,'_06')
    reshape6=cnnHAR.inference_cov21(pool16,'_06')
    local26=cnnHAR.inference_local21(reshape6,'_06')
    local36=cnnHAR.inference_local31(local26,'_06')
    local46=cnnHAR.inference_local41(local36,'_06')
    logits6=cnnHAR.inference_output1(local46,'_06')
    
    loss1=cnnHAR.loss(logits1, labels,'_01')
    loss2=cnnHAR.loss(logits2, labels,'_02')
    loss3=cnnHAR.loss(logits3, labels,'_03')
    loss4=cnnHAR.loss(logits4, labels,'_04')
    loss5=cnnHAR.loss(logits5, labels,'_05')
    loss6=cnnHAR.loss(logits6, labels,'_06')
  
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cnnHAR.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    #print('!!!!!!!!!!!!!!!!!!!variables to restore:')
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver,summary_writer,labels,loss1,logits1,loss2,logits2,loss3,logits3,loss4,logits4,loss5,logits5,loss6,logits6,summary_op)
      #loss7,logits7,loss8,logits8,loss9,logits9,loss10,logits10,loss11,logits11,loss12,logits12
      if FLAGS.run_once:
        break
      #time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
