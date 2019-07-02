report_s01 = False
report_s02 = False
report_s03 = False
report_s04 = False
report_s05 = False
report_s06 = False
report_s07 = False

vectorizationLength = 20

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# The maximum length sentence we want for a single input in characters
seq_length = 100

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

#Execute the training¶
EPOCHS=3

# Number of characters to generate
num_generate = 1000

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0