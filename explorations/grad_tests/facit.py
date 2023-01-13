import tensorflow as tf
import numpy as np


sigmoid = lambda x : 1 / (1 + tf.exp(-x))

with tf.GradientTape() as tape:
    cost = tf.Variable(0.0, 'float32')
    A = tf.Variable([
        [.3, -.2],
        [-.1, .5]
    ], 'float32')
    B = tf.Variable([
        [.42, -.76],
        [.37, -.12]
    ], 'float32')
    inp = tf.Variable([.1, -.3])
    neurons = [inp[:]]
    layers = [(A, tf.zeros(2, 'float32')), (B, tf.zeros(2, 'float32'))]

    for layer in layers:
        M, b = layer
        inp = tf.matmul([inp], M) + [b]
        inp = inp[0, :]
        neurons.append(inp)

    cost.assign_add(inp[0]*inp[0])
    cost.assign_add((inp[0]-1)*(inp[0]-1))
    cost.assign(cost / 2)
    
grad = tape.gradient(cost, [A, B, neurons])

print(grad)


    


