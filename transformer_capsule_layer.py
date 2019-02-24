"""Author: Brandon Trabucco, Copyright 2019
Implements a convolutional capsule layer with a scaled dot product attention mechanism.
MIT License
"""


import tensorflow as tf


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, hidden_size, output_size, **kwargs):
        super(ScaledDotProductAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.key_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.value_map = tf.keras.layers.Dense(hidden_size * num_heads)
        self.output_map = tf.keras.layers.Dense(output_size)
    
    def __call__(self, queries, keys, values):
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.query_map(queries), self.key_map(keys), self.value_map(values)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_size])
        return self.output_map(S)  
        
    @property
    def trainable_variables(self):
        layer_variables = (
            self.query_map.trainable_variables + self.key_map.trainable_variables + 
            self.value_map.trainable_variables + self.output_map.trainable_variables )
        return layer_variables
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        layer_variables = (
            self.query_map.variables + self.key_map.variables + 
            self.value_map.variables + self.output_map.variables )
        return layer_variables
    
    @property
    def weights(self):
        return self.variables


def capsule_activation_function(x):
    capsule_norm = tf.norm(x, axis=-1, keepdims=True)
    return ((capsule_norm * capsule_norm / (1 + capsule_norm * capsule_norm)) * 
        x / capsule_norm)


class CapsuleLayer(tf.keras.layers.Layer):

    def __init__(self, num_capsules, capsule_size, num_heads, kernel_size=(3, 3), strides=(1, 1), padding="same"):
        self.conv_map = tf.keras.layers.Conv2D(num_capsules * capsule_size, kernel_size, 
            strides=strides, padding=padding)
        self.attention_map = ScaledDotProductAttentionLayer(num_heads, capsule_size // num_heads, capsule_size)
        self.num_capsules = num_capsules
        self.capsule_size = capsule_size

    def __call__(self, inputs):
        X = self.conv_map(inputs)
        batch_size, height, width = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2])
        X = tf.reshape(tf.reshape(X, [-1, self.num_capsules * self.capsule_size]), 
            [-1, self.num_capsules, self.capsule_size])
        X = capsule_activation_function(X)
        X = capsule_activation_function(X + self.attention_map(X, X, X))
        return tf.reshape(tf.reshape(X, [-1, self.num_capsules * self.capsule_size]), 
            [batch_size, height, width, self.num_capsules * self.capsule_size])
        
    @property
    def trainable_variables(self):
        return (self.conv_map.trainable_variables + self.attention_map.trainable_variables)
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return (self.conv_map.variables + self.attention_map.variables)
    
    @property
    def weights(self):
        return self.variables


if __name__ == "__main__":

    image = tf.random_normal([100, 64, 64, 3])
    layer_one = CapsuleLayer(16, 16, 4)
    z = layer_one(image)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1"))) as sess:

        sess.run(tf.global_variables_initializer())
        print(sess.run(z).shape)
        print("Finished test.")
