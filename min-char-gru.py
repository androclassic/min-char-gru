"""
Minimal character-level using GRU based on Vanilla RNN model written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import os, sys
from random import uniform

def sigmoid(x):
 return (1 / (1 + np.exp(-x)))

class GRUModel:
  def __init__(self, input_size, hidden_size):
  # model parameters
    self.Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Wxg = np.random.randn(hidden_size, input_size)*0.01 # input to candidate gate
    self.Whg = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to to candidate gate
    self.Wxf = np.random.randn(hidden_size, input_size)*0.01 # input to forget gate
    self.Whf = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to to forget gate
    self.Why = np.random.randn(input_size, hidden_size)*0.01 # hidden to output
    self.bh =  np.zeros((hidden_size, 1)) # hidden bias
    self.bg =  np.zeros((hidden_size, 1)) # gate bias
    self.bf =  np.zeros((hidden_size, 1)) # forget gate bias
    self.by =  np.zeros((input_size, 1)) # output bias
    self.input_size = input_size


    self.mWxh, self.mWhh, self.mWxg, self.mWhg, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Wxg), np.zeros_like(self.Whg), np.zeros_like(self.Why)
    self.mWxf, self.mWhf = np.zeros_like(self.Wxf), np.zeros_like(self.Whf)

    self.mbh, self.mbg, self.mby = np.zeros_like(self.bh), np.zeros_like(self.bg) , np.zeros_like(self.by) # memory variables for Adagrad
    self.mbf = np.zeros_like(self.bf)

  def save_model(self,folder_name):
    if os.path.exists(folder_name) == False:
      os.mkdir(folder_name)

    for param, name in zip([self.Wxh, self.Whh, self.Wxg, self.Whg, self.Wxf, self.Whf, self.Why, self.bh, self.bg, self.bf, self.by], 
                           ['Wxh', 'Whh','Wxg', 'Whg','Wxf', 'Whf', 'Why', 'bh', 'bg', 'bf', 'by']):
      np.save(folder_name+"/"+ name, param)

  def load_model(self, folder_name):
    for param, name in zip([self.Wxh, self.Whh, self.Wxg, self.Whg, self.Wxf, self.Whf, self.Why, self.bh, self.bg, self.bf, self.by], 
                           ['Wxh', 'Whh','Wxg', 'Whg','Wxf', 'Whf', 'Why', 'bh', 'bg', 'bf', 'by']):

      param *= 0 #using this trick I make sure there is the same reference for the params I want to load
      param += np.load(folder_name+"/"+ name +".npy")

  def _froward_pass(self, inputs, hprev):
    xs, hs, hc, gs, fs, ys, ps = {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    # forward pass
    for t in range(len(inputs)):
      xs[t] = np.zeros((self.input_size,1)) # encode in 1-of-k representation
      xs[t][inputs[t]] = 1
      fs[t] = sigmoid(np.dot(self.Wxf, xs[t]) + np.dot(self.Whf, hs[t-1]) + self.bf) #gate state
      hc[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, fs[t] *  hs[t-1]) + self.bh) # hidden state candidate
      gs[t] = sigmoid(np.dot(self.Wxg, xs[t]) + np.dot(self.Whg, hs[t-1]) + self.bg) #gate state
      hs[t] = gs[t] * hc[t] + (1-gs[t]) * hs[t-1]
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars

    return xs, hs, hc, gs, fs, ys, ps 

  def forward_pass(self, inputs):
    _, _, _, _, _, _, ps =  self._froward_pass(inputs)
    return ps 


  def compute_loss(self, targets, pred):
    # softmax (cross-entropy loss
    loss = 0
    for t in range(len(targets)):
      loss += -np.log(pred[t][targets[t],0]) # softmax (cross-entropy loss
    return loss


  def lossFun(self, inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, hc, gs, fs, ys, ps = self._froward_pass(inputs, hprev)
    loss = self.compute_loss(targets, ps)


    # backward pass: compute gradients going backwards
    dWxh, dWhh,dWxg, dWhg, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh),np.zeros_like(self.Wxg), np.zeros_like(self.Whg), np.zeros_like(self.Why)
    dWxf, dWhf = np.zeros_like(self.Wxf), np.zeros_like(self.Whf)
    dbh, dbg, dbf, dby = np.zeros_like(self.bh), np.zeros_like(self.bg), np.zeros_like(self.bf), np.zeros_like(self.by)
 
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
      dy = np.copy(ps[t])
      dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      dg = (hc[t] - hs[t-1]) * dh
      dgraw = gs[t] * (1-gs[t]) * dg
      dWxg += np.dot(dgraw, xs[t].T)
      dWhg += np.dot(dgraw, hs[t-1].T)
      dbg += dgraw

      dhc = gs[t] * dh 
      dhcraw = (1 - hc[t] * hc[t]) * dhc  # backprop through tanh nonlinearity
      dbh += dhcraw
      dWxh += np.dot(dhcraw, xs[t].T)
      hf = fs[t] *  hs[t-1]
      dWhh += np.dot(dhcraw, hf.T)

      dfs =  np.dot(self.Whh.T, dhcraw) * hs[t-1]
      dfraw = fs[t] * (1-fs[t]) * dfs
      dWxf += np.dot(dfraw, xs[t].T)
      dWhf += np.dot(dfraw, hs[t-1].T)
      dbf += dfraw

      dhnext =  np.dot(self.Whh.T, dhcraw) * fs[t]  + np.dot(self.Whg.T, dgraw) + np.dot(self.Whf.T, dfraw) + (1-gs[t]) * dh 

    for dparam in [dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby, hs[len(inputs)-1]


  def optimize(self, dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby):
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Wxg, self.Whg, self.Wxf, self.Whf, self.Why, self.bh, self.bg, self.bf, self.by], 
                                  [dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby], 
                                  [self.mWxh, self.mWhh, self.mWxg, self.mWhg, self.mWxf, self.mWhf, self.mWhy, self.mbh, self.mbg, self.mbf, self.mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update


  # gradient checking
  def gradCheck(self, inputs, targets, hprev):
    num_checks, delta = 5, 1e-5
    _, dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby, _ = self.lossFun(inputs, targets, hprev)

    print("****************************************************************")

    for param, dparam, name in zip([self.Wxh, self.Whh, self.Wxg, self.Whg, self.Wxf, self.Whf, self.Why, self.bh, self.bg, self.bf, self.by], 
                                   [dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby], 
                                   ['Wxh', 'Whh','Wxg', 'Whg','Wxf', 'Whf', 'Why', 'bh', 'bg', 'bf', 'by']):

      s0 = dparam.shape
      s1 = param.shape
  #    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
      print(name)
      for i in range(num_checks):
        ri = int(uniform(0,param.size))
        while dparam.flat[ri] == 0:
          ri = int(uniform(0,param.size))
    
        # evaluate cost at [x + delta] and [x - delta]
        old_val = param.flat[ri]
        param.flat[ri] = old_val + delta
        cg0, _, _, _, _, _, _, _, _, _, _, _, _ = self.lossFun(inputs, targets, hprev)
        param.flat[ri] = old_val - delta
        cg1, _, _, _, _, _, _, _, _, _, _, _, _ = self.lossFun(inputs, targets, hprev)
        param.flat[ri] = old_val # reset old value for this parameter
        # fetch both numerical and analytic gradient
        grad_analytic = dparam.flat[ri]
        grad_numerical = (cg0 - cg1) / ( 2 * delta )
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
        # rel_error should be on order of 1e-7 or less

  def sample(self, h, seed_ix, n, max_prob = False):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((self.input_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
      f = sigmoid(np.dot(self.Wxf, x) + np.dot(self.Whf, h) + self.bf) #forget gate state
      hc = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh,f * h) + self.bh) # hidden state candidate
      g = sigmoid(np.dot(self.Wxg, x) + np.dot(self.Whg, h) + self.bg) #gate state
      h = g * hc + (1-g) * h
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      if max_prob :
        ix = p.argmax()
      else :
        ix = np.random.choice(range(self.input_size), p=p.ravel())
      x = np.zeros((self.input_size, 1))
      x[ix] = 1
      ixes.append(ix)
    return ixes



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



model = GRUModel(vocab_size, hidden_size)

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


  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby, hprev = model.lossFun(inputs, targets, hprev)
#  model.gradCheck(inputs, targets, hprev)


  smooth_loss = smooth_loss * 0.999 + loss * 0.001
#  smooth_loss = loss
  if n % 300 == 0: print('iter %d, loss: %f instant_loss: %f' % (n, smooth_loss, loss)) # print progress

  model.optimize(dWxh, dWhh, dWxg, dWhg, dWxf, dWhf, dWhy, dbh, dbg, dbf, dby)


  p += seq_length # move data pointer
  n += 1 # iteration counter 
