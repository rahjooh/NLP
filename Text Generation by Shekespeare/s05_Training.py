import s03_DataPrepration as s3,s04_Modeling as s4 ,conf as c
import tensorflow as tf , os

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss  = loss(s4.target_example_batch, s4.example_batch_predictions)
if c.report_s05 :
    print("Prediction shape: ", s4.example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

s4.model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints1'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#Configure checkpoints
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


history = s4.model.fit(s3.dataset.repeat(), epochs=c.EPOCHS, steps_per_epoch=s3.steps_per_epoch, callbacks=[checkpoint_callback])