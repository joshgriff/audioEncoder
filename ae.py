# Auto encoder


from keras import Sequential, Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import Input, concatenate, Embedding, LSTM
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator

import numpy as np

import matplotlib.pyplot as plt

# Let's start with something easy

l = 10

s1 = np.linspace(1,100,100)
s1 -= s1.min()
s1 = s1/s1.max()


s = s1
for i in range(1,1000):
	s = np.append(s,s1)

def windowGen(x,length):
		l = x.shape[0]
		for i in range(l-length):
			idx = [ix for ix in range(i,i+length)]
			yield(np.take(x,idx,axis=0))

# Generator Zipper 1,1 -> (1,1)
def xygen(xg,yg):
	for x,y in zip(xg,yg):
		yield(x[None,:],y[None,:])


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

model = autoencoder
model.compile(optimizer='adagrad')

X_train = windowGen(s,l)
Y_train = windowGen(s,l)

gen = xygen(X_train,Y_train)

spe = len(s)-l
epochs = 1

model.fit_generator(gen,steps_per_epoch=spe,epochs=epochs)

st = s1[:10]
so = model.predict(st[None,:])

plt.figure(1)
plt.plot(so[0])

plt.figure(2)
plt.plot(st)

plt.show()