import  s02_vectorization as s2 ,s04_Modeling as s4 ,conf as c
import tensorflow as tf
import os , time
# Directory where the checkpoints was saved
checkpoint_dir = './training_checkpoints1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

tf.train.latest_checkpoint(checkpoint_dir)

model = s4.build_model(s4.vocab_size, c.embedding_dim, c.rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model, start_string):
    print('#### generate_text ####')
    # Evaluation step (generating text using the learned model)

    # Converting our start string to numbers (vectorizing)
    input_eval = [s2.char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(c.num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / c.temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(s2.idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

text1 =generate_text(model, start_string=u"ROMEO: ")
print(text1)
with open("Output s6 "+time.strftime("%Y%m%d%H%M%S")+".txt", "w") as text_file:
    print(f"Purchase Amount: {text1}", file=text_file)