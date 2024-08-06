import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Dense
import math 
from lab_coffee_utils import load_coffee_data

x,y=load_coffee_data()



norm_layer=tf.keras.layers.Normalization(axis=1)
norm_layer.adapt(x)
x_normalized=norm_layer(x)
# print(f"max temp , min temp \n{np.max(x_normalized[:,0]):0.3f} , {np.min(x_normalized[:,1]):0.3f}")
tf.random.set_seed(124)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(units=3 , activation='sigmoid', name="layer_1"),
    Dense(units=1 , activation='sigmoid', name="layer_2")
])
Xt=np.tile(x,(1000,1))
Yt=np.tile(y,(1000,1))
model.summary()

#print(f"value of w's and b's are {model.get_layer("layer_1").get_weights()}")
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

#print(y.shape)


model.fit(
    Xt,Yt,            
    epochs=5,
)

test_array=np.array([
    [200,13.9],
    [200,17]
])
test_array_norm=norm_layer(test_array)
prediction = model.predict(test_array_norm)
yHat= (prediction>=0.5).astype(int)
print(yHat)