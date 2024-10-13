
import os
os.environ['KERAS_BACKEND'] = 'torch'

import cProfile
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Embedding, Reshape
import tensorflow as tf
import time

def make_one_hot(n, i):
    return [int(i == j) for j in range(n)]

def main():
    input_tensor_size = 500
    output_tensor_size = 6
    model = Sequential([
        InputLayer( (input_tensor_size,) ),
        #Embedding(observation_count, 4),
        #Reshape((4,)),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(output_tensor_size, activation='linear'),
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    #model.build(make_input_tensor(0).shape)
    model.summary()
    #help(model.__call__)
    #print(model.__call__)

    for i in range(100):
        before = time.perf_counter()
        input_tensor = np.random.random( (1,input_tensor_size) )
        #print(input_tensor)
        futures = model.predict(input_tensor, verbose=0, batch_size=1)#.numpy()
        #futures = model(input_tensor).numpy()
        print("timing", int((time.perf_counter() - before)*1000))

try:
    #cProfile.run("main()", sort="cumtime")
    main()
except KeyboardInterrupt:
    pass
