import sys
sys.path.insert(1,'../')
import textCorpus.brown as brown

# %matplotlib inline
# import math
# from d2l import tensorflow as d2l

import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential



import keras
# from tensorflow.keras import keras
# from keras import ops
from keras import layers



class Linear(keras.layers.Layer):
    def __init__(self,units=32, input_shape=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_shape,units),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(shape=(units,),initializer="zeros",trainable=True)

    def call(self,inputs):
        # resource - https://www.scaler.com/topics/keras/custom-layers-in-keras/
        # resource - https://keras.io/guides/making_new_layers_and_models_via_subclassing/

        # need to return ops.matmul(inputs,self.w) + self.b
        return tf.matmul(inputs, self.w) + self.b


class Model(keras.Model):

    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.layer_1 = Linear(300,input_shape = vocabulary_size)
        self.layer_2 = Linear(vocabulary_size,input_shape=300)
        self.layer_3 = Linear(1,input_shape=vocabulary_size)

    def call(self,inputs):
        x = self.layer_1(inputs)
        x = keras.activations.relu(x)
        x = self.layer_2(x)
        x = self.layer_3(x)


        return self.layer_1
    

dataset_tokens, mapping, reverse_mapping = brown.dataset()

vocabulary_size = len(mapping)  # Adjust based on your vocabulary size
embedding_dim = len(dataset_tokens)  # Adjust based on your preference
hidden_units = 300  # Set to 300 for the Brown Corpus model

print(vocabulary_size,embedding_dim)

model = Model(23,1)
model.build((1,vocabulary_size))
print(model.summary())





#need pydot and graphviz
# keras.utils.plot_model(model,to_file="model.png")




















