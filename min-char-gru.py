"""
Minimal character-level using GRU based on Vanilla RNN model written by Andrej Karpathy (@karpathy)
BSD License
"""
import gru_model as gru
import numpy as np
import os, sys

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 75 # number of steps to unroll the RNN for
learning_rate = 1e-2 * 7 
saving_step = 10000
train_step = 5
idx = -1

folder_name = "model_"+str(hidden_size)
if os.path.exists(folder_name) == False:
  os.mkdir(folder_name)


if idx != -1:
  ix_to_char =   np.load(folder_name+"/ix_to_char.npy", allow_pickle=True).tolist()
  char_to_ix =   np.load(folder_name+"/char_to_ix.npy", allow_pickle=True).tolist()
  chars =   np.load(folder_name+"/chars.npy",allow_pickle=True).tolist()
else:
  chars = list(set(data))
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }
  np.save(folder_name+"/ix_to_char",ix_to_char)
  np.save(folder_name+"/char_to_ix",char_to_ix)
  np.save(folder_name+"/chars",chars)

data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))
print(ix_to_char)

model = gru.GRUModel(vocab_size, hidden_size)

if idx != -1:
  model.load_model(folder_name+"/"+str(idx))

n, p = 0, 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

inputs = []
targets = []
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data

  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 1000 == 0:                          
    sample_ix = model.sample(hprev, inputs[0], 500)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  if n % saving_step == 0:
    name = folder_name+"/"+str(int(n/saving_step))
    model.save_model(name)

  xs = {}
  for t in range(len(inputs)): 
      xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1

  # forward seq_length characters through the net and fetch gradient
  loss, params, hprev = model.lossFun(xs, targets, hprev)
#  model.gradCheck(inputs, targets, hprev)

  smooth_loss = smooth_loss * 0.999 + loss * 0.001
#  smooth_loss = loss
  if n % 300 == 0: print('iter %d, loss: %f instant_loss: %f' % (n, smooth_loss, loss)) # print progress

  model.optimize(params, learning_rate)

  p += seq_length # move data pointer
  n += 1 # iteration counter 
