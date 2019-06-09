import gru_model as gru
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import math

hidden_size = 25 # size of hidden layer of neurons
seq_length = 15 # number of steps to unroll the RNN for
offset  = 5 #number of steps ahead we want to predict
learning_rate = 1e-2 * 7  
train_size = 1000.0
nb_iterations = 10000

def foo(x):
	return np.sin(x) * 0.1 + np.sin(x/2) * 0.05 + np.sin(x*2.3) * 0.25 + np.tanh(x* 1.4) * 0.25 + np.cos(x*2.4) * 0.35 


#normalized sequential values
x_train = np.arange(train_size) / 10
y_train = foo(x_train)
model = gru.GRUModel(1, hidden_size)

n, p = 0, 0,
smooth_loss = -np.log(1.0)*seq_length # loss at iteration 0

inputs = []
targets = []
while n < nb_iterations:
  p = np.random.choice(range(len(x_train)-  seq_length - offset))

  inputs = foo(x_train[p:p+seq_length])
  targets = foo(x_train[p+offset :p+seq_length+offset])

  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  # forward seq_length characters through the net and fetch gradient
  loss, params, hprev = model.lossFun(inputs, targets, hprev, loss_function='rmse_last')
#  model.gradCheck(inputs, targets, hprev)

  smooth_loss = smooth_loss * 0.999 + loss * 0.001
#  smooth_loss = loss
  if n % 300 == 0: print('iter %d, loss: %f instant_loss: %f' % (n, smooth_loss, loss)) # print progress

  model.optimize(params, learning_rate)
  n += 1 # iteration counter 


x_test = np.arange(train_size,2*train_size) / 10.0
y_test = []
for p in range(0, len(x_test)):
	hprev = np.zeros((hidden_size,1)) # reset RNN memory
	inputs = foo(x_test[p:p+seq_length])
	result = model.forward_pass(inputs, hprev)
	result = result[len(result) - 1].flatten()	
	y_test.append(result)

plt.subplot(211)
plt.plot(x_train, y_train)
plt.subplot(212)
plt.plot(x_test, foo(x_test))
plt.plot(x_test + (seq_length+offset) / 10.0, y_test)

plt.show()