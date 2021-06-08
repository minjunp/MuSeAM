import tensorflow as tf
import numpy
import numpy as np

@tf.function()
def true_rank_loss(yTrue, yPred):
  ##### get vector ranks
  rank_yTrue = tf.argsort(tf.argsort(yTrue))
  rank_yPred = tf.argsort(tf.argsort(yPred))

  #print(f'[INFO] Print ranked yTrue: {rank_yTrue}')
  #print(f'[INFO] Print ranked yPred: {rank_yPred}')

  ##### pass to numpy
  rank_yTrue = rank_yTrue.numpy().flatten()
  rank_yPred = rank_yPred.numpy().flatten()

  ##### create ranks tuple
  ranks_tuple = []
  index=0
  for index in range(len(rank_yTrue)):

    ranks_tuple.append((rank_yTrue[index],rank_yPred[index]))
    index+=1
  #print(f'[INFO] Print ranked tuple: {ranks_tuple}')

  ##### sort tuple by rank_yPred
  def tuple_sort(my_tuple):
    return(sorted(my_tuple, key = lambda x: x[1]))

  sorted_ranks_tuple = tuple_sort(ranks_tuple)
  #print(f'[INFO] Print sorted ranked tuple by rank_yPred: {sorted_ranks_tuple}')

  ##### get sorted rank vector
  sorted_rank_yTrue = [i[0] for i in sorted_ranks_tuple]
  #print(f'[INFO] Print sorted ranked of rank_yTrue: {sorted_rank_yTrue}')

  ##### calculate loss
  sorted_rank_yTrue = np.array(sorted_rank_yTrue)
  diag = np.tile(sorted_rank_yTrue, (sorted_rank_yTrue.shape[0],1))
  loss = np.triu(diag-sorted_rank_yTrue.reshape(-1,1))<0
  #print(f'[INFO] Print loss matrix: \n {loss}')
  loss = np.sum(loss)
  #print(f'[INFO] Print final loss: {loss}')
  loss = tf.cast(loss,dtype="float32")
  return loss

#Create new loss function (Rank mse)
@tf.function()
def rank_mse(yTrue, yPred):
  lambda_value=0.25
  #pass lambda value as tensor
  lambda_value = tf.convert_to_tensor(lambda_value,dtype="float32")

  #get vector ranks
  rank_yTrue = tf.argsort(tf.argsort(yTrue))
  rank_yPred = tf.argsort(tf.argsort(yPred))

  #calculate losses
  mse = tf.reduce_mean(tf.square(tf.subtract(yTrue,yPred)))
  rank_mse = tf.reduce_mean(tf.square(tf.subtract(rank_yTrue,rank_yPred)))

  #take everything to same dtype
  mse = tf.cast(mse,dtype="float32")
  rank_mse = tf.cast(rank_mse,dtype="float32")

  #(1 - lambda value)* mse(part a of loss)
  loss_a = tf.multiply(tf.subtract(tf.ones(1,dtype="float32"),lambda_value),mse)
  #lambda value * rank_mse (part b of loss)
  loss_b = tf.multiply(lambda_value,rank_mse)
  #final loss
  loss = tf.add(loss_a,loss_b)

  return loss
