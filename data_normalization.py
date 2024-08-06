import numpy as np
import tensorflow as tf

import math

from lab_coffee_utils import load_coffee_data

xArr,yArr=load_coffee_data()
#print(xArr)
print(tf.__version__)
## MAx and min Temperature before normalization and after 
print(f"Max Temp before Normalization : {np.max(xArr[:,0]):0.2f}")
print(f"Min Temp before Normalization : {np.min(xArr[:,0]):0.2f}")
print(f"Max Duration before Normalization : {np.max(xArr[:,1]):0.2f}")
print(f"Min Duration before Normalization : {np.min(xArr[:,1]):0.2f}")


def calc_mean (arrr) :
    m=arrr.shape[0]
    sum=0
    for i in range (m) :
        sum+=arrr[i,0]
    sum=sum/m;
    return sum

def calc_mean_Duration (arrr) :
    m=arrr.shape[0]
    sum=0
    for i in range (m) :
        sum+=arrr[i,1]
    sum=sum/m;
    return sum


def standard_deviation(mean,xArr) :
    m=xArr.shape[0]
    sum=0
    for i in  range (m) :
        x_i=xArr[i,0]-mean
        x_i=x_i**2
        sum+=x_i
    sum=sum/(m-1)
    std=math.sqrt(sum)
    return std 
def standard_deviationDuration(mean,xArr) :
    m=xArr.shape[0]
    sum=0
    for i in  range (m) :
        x_i=xArr[i,1]-mean
        x_i=x_i**2
        sum+=x_i
    sum=sum/(m-1)
    std=math.sqrt(sum)
    return std 

def calc_norm_temp( arr) :
    mean = calc_mean(arr)
    std = standard_deviation(mean, arr)
    arr= ( arr - mean ) / std
    return arr


def calc_norm_duration( arr) :
    mean = calc_mean_Duration(arr) # mean 13 aaya hai jo ke blkul sahi hai
    std = standard_deviationDuration(mean, arr)  # std = 1.13
    arr= ( arr[:,1] - mean ) / std 
    return arr


xArr_Temp = calc_norm_temp(xArr)
xArr_Duration= calc_norm_duration(xArr)
print(f"Max Temp After Normalization : {np.max(xArr_Temp[:,0]):0.2f}")
print(f"Min Temp After Normalization : {np.min(xArr_Temp[:,0]):0.2f}")
print(f"Max Duration After Normalization : {np.max(xArr_Duration):0.2f}")
print(f"Min Duration After Normalization : {np.min(xArr_Duration):0.2f}")


