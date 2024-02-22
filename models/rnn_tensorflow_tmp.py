import tensorflow as tf
print(tf.__version__)

import keras

class Layer(keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="random_normal",
                                 trainable=True)
        
    def call(self, inputs):
        return tf.matmul(inputs,self.w) + self.b
    

class Model(keras.Model):
    def __init__(self, input_size , embedding_size, hidden_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = Layer(embedding_size)
        self.weight = Layer(hidden_size)
        self.u = Layer(hidden_size)
        self.v = Layer(output_size)

    def call(self,input,hidden_state):
        e = self.embedding(input)
        print(e.shape)
        h1 = self.weight(e)
        h2 = self.u(hidden_state)
        h = tf.math.add(h1,h2)
        out = self.v(h)
        out = tf.nn.softmax(out)
        # out = tf.reshape(out)
        tf.reshape(out,-1)

        return out,h

