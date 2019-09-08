import keras
import numpy as np

x = np.array([0,1,2,3,4])
y = 2*x + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1,input_shape=(1,)))

model.compile('SGD', 'mse')
model.fit(x[:2], y[:2], epochs=100, verbose=0)

print(y[2:])
print(model.predict(x[2:]).flatten())
