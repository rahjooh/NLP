save_dir = 'models/new_save'

data_dir = 'data/scotus'

#number of cells per block
block_size = 2048 


num_blocks=3 #number of blocks per layer
num_layers=3 #number of layers
model = 'gru' #'rnn, gru, lstm or nas
batch_size = 40 #'minibatch size
seq_length=40 #RNN sequence length
num_epochs=50 #number of epochs
save_every=5000 #save frequency
grad_clip =5.#clip gradients at this value
learning_rate=1e-5 #learning rate
decay_rate = 0.975 #how much to decay the learning rate
decay_steps=100000 #how often to decay the learning rate


