import tensorflow as tf
import numpy as np
import numpy
import os
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#Please set tf.config.run_functions_eagerly(True) before using this loss function
@tf.function()
def rank_mse(yTrue, yPred):

  def calculate_loss(yTrue, yPred):
    yTrue = tf.reshape(yTrue,shape=(1,yTrue.shape[0]))
    yPred = tf.reshape(yPred,shape=(1,yPred.shape[0]))
    #do
    #lambda_value=0.5
    lambda_value=0.2
    size = yTrue.get_shape()[1]
    #pass lambda value as tensor
    lambda_value = tf.convert_to_tensor(lambda_value,dtype="float32")
    #get vector ranks
    rank_yTrue = tf.argsort(tf.argsort(yTrue))
    rank_yPred = tf.argsort(tf.argsort(yPred))

    print(f'yTrue values are {yTrue}')
    print(f'yPred values are {yPred}')

    print(f'[INFO] Print ranked yTrue: {rank_yTrue}')
    print(f'[INFO] Print ranked yPred: {rank_yPred}')
    #calculate losses

    #calculate mse
    print(f'\n[INFO] Calculating normal mse')
    mse = tf.subtract(yTrue,yPred)
    print(f'[INFO] subtract mse: {mse}')
    mse = tf.square(mse)
    print(f'[INFO] square mse: {mse}')
    mse = tf.math.reduce_sum(mse).numpy()
    print(f'[INFO] reduce sum mse: {mse}')
    mse = tf.divide(mse,size)
    print(f'[INFO] divide by size mse: {mse}')
    mse = tf.cast(mse,dtype="float32")
    print(f'[INFO] final mse: {mse}')

    #calculate rank_mse
    print(f'\n[INFO] Calculating rank mse')
    rank_mse = tf.cast(tf.subtract(rank_yTrue,rank_yPred),dtype="float32")
    print(f'[INFO] substract rank_mse: {rank_mse}')
    rank_mse = tf.square(rank_mse)
    print(f'[INFO] square rank_mse: {rank_mse}')
    rank_mse = tf.math.reduce_sum(rank_mse).numpy()
    print(f'[INFO] reduce sum rank_mse: {rank_mse}')
    rank_mse = tf.math.sqrt(rank_mse)
    print(f'[INFO] square root rank_mse: {rank_mse}')
    rank_mse = tf.divide(rank_mse,size)
    print(f'[INFO] divide by size rank_mse: {rank_mse}')
    print(f'[INFO] final rank_mse: {rank_mse}')

    #(1 - lambda value)* mse(part a of loss)
    loss_a = tf.multiply(tf.subtract(tf.ones(1,dtype="float32"),lambda_value),mse)
    print(f'loss_a is {loss_a}')
    #lambda value * rank_mse (part b of loss)
    loss_b = tf.multiply(lambda_value,rank_mse)
    print(f'loss_b is {loss_b}')
    #final loss
    loss = tf.add(loss_a,loss_b)
    return loss

  debug=False

  if not debug:
    with HiddenPrints():
      loss = calculate_loss(yTrue, yPred)
      return loss
  else:
    loss = calculate_loss(yTrue, yPred)
    return loss



@tf.function()
def mean_true_rank_loss(yTrue, yPred):
  #get vector ranks
  rank_yTrue = tf.argsort(tf.argsort(yTrue))
  rank_yPred = tf.argsort(tf.argsort(yPred))

  #print(f'[INFO] Print ranked yTrue: {rank_yTrue}')
  #print(f'[INFO] Print ranked yPred: {rank_yPred}')

  #pass to numpy
  rank_yTrue = rank_yTrue.numpy().flatten()
  rank_yPred = rank_yPred.numpy().flatten()

  #create ranks tuple
  ranks_tuple = []
  index=0
  for index in range(len(rank_yTrue)):

    ranks_tuple.append((rank_yTrue[index],rank_yPred[index]))
    index+=1
  #print(f'[INFO] Print ranked tuple: {ranks_tuple}')

  #sort tuple by rank_yPred
  def tuple_sort(my_tuple):
    return(sorted(my_tuple, key = lambda x: x[1]))

  sorted_ranks_tuple = tuple_sort(ranks_tuple)
  #print(f'[INFO] Print sorted ranked tuple by rank_yPred: {sorted_ranks_tuple}')

  #get sorted rank vector
  sorted_rank_yTrue = [i[0] for i in sorted_ranks_tuple]
  #print(f'[INFO] Print sorted ranked of rank_yTrue: {sorted_rank_yTrue}')

  #calculate loss
  sorted_rank_yTrue = np.array(sorted_rank_yTrue)
  diag = np.tile(sorted_rank_yTrue, (sorted_rank_yTrue.shape[0],1))
  loss = np.triu(diag-sorted_rank_yTrue.reshape(-1,1))<0
  #print(f'[INFO] Print loss matrix: \n {loss}')
  loss = np.sum(loss)
  #print(f'[INFO] Print loss: {loss}')
  loss = tf.cast(loss,dtype="float32")
  loss = tf.divide(loss,yTrue.get_shape()[1])
  #print(f'[INFO] Print loss divided by mean: {loss}')
  return loss
