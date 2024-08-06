import numpy as np
import tensorflow as tf
from lab_coffee_utils import load_coffee_data

x,y=load_coffee_data()
norm_l=tf.keras.layers.Normalization(axis=1) # can also use axis =-1 
norm_l.adapt(x) # this learns mean and variance
Xn=norm_l(x) # this normalizes the data
# print(f"max and min temp post normalization : {np.max(Xn[:,0]):0.2f} , {np.min(Xn[:,0]):0.2f}")



sample_arr=np.array([[100,2],[200,4],[300,6]])
print (" OUR ARRAY BEFORE USING THE TILE FUNCTION \n\n",sample_arr)
sample_arr_tile=np.tile(sample_arr,(5,2))
print ("\n OUR ARRAY AFTER USING THE TILE FUNCTION WITH (5,2) Array is repeated 5 times in Rows , and 2 times in columns\n")
print(sample_arr_tile)

