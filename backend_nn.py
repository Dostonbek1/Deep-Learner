import keras
import time
from pathlib import *
import subprocess
import pandas as pd
import rpy2.robjects as robjects
import numpy as np
from keras.callbacks import History
import matplotlib.pyplot as plt
import tkinter as tk


def make_prep():
    
    subprocess.call(["R the_recepie_scrip_imputedt.R"], shell=True)

def n_network(data,optimizer_var, layers, batch_sz, epochs_size, regression_status,validation_spliter):
    global num_cat
    multi_class = False
    metrics_value = 'accuracy'
    if num_cat != 1:
        multi_class = True
    if multi_class == False and not regression_status:
        activation_type = 'sigmoid'
        loss_func = 'binary_crossentropy'
        output_layer = 1
        y_train = data[:, 1]
    
    else:
        activation_type = 'softmax'
        loss_func = 'categorical_crossentropy'
        num_cat += 1
        y_train=keras.utils.to_categorical(data[:, 1],num_classes=num_cat) 
        
    if regression_status:
        loss_func='mean_squared_error'
        y_train = y_trainers
        metrics_value = 'mae'

    print(type(y_train))
    x_train = data[:, 2:]


    model=keras.models.Sequential()
    
    if len(layers) > 0:
        for layer in layers:
            model.add(keras.layers.Dense(layer[0],activation="relu"))

    if regression_status:
        model.add(keras.layers.Dense(1))
    else:
        model.add(keras.layers.Dense(num_cat,activation=activation_type))


    model.compile(optimizer=optimizer_var,
                  loss=loss_func,
                  metrics=[metrics_value],
                  )
    hist=History()
    model.fit(x_train,y_train,batch_sz,epochs_size,validation_split=validation_spliter,callbacks=[hist])
    return model, hist

def ploter(hist):
    hist_keys=list(hist.history.keys())
    print("hist keys",hist_keys)
    plt.plot(hist.history[hist_keys[1]],'ro-', label=hist_keys[1])
    plt.plot(hist.history[hist_keys[3]],'go-', label=hist_keys[3])
    plt.title('Model Accuracy')
    plt.ylabel(hist_keys[3])
    plt.xlabel('epoch')
    plt.legend()
    plt.show(block=True)
    time.sleep(5)
    plt.close("all")

def save(model):
    model_str=model.to_json()
    f_path=Path("output/model_str.json")
    f_path.write_text(model_str)
    model.save_weights("output/model_weights.h5")

def loader(target,path,dummy_input,reg_val):
    dummy_bool = '#'
    if dummy_input != False:
        dummy_bool = ''

    robjects.r(r'''
            install.packages("recipes")
            memory.limit(10000)
            library(recipes)
            data_set <- read.csv("{0}")
            data_set<-data_set[-1]
            data_set <- data_set %>% select({1}, everything())
            recepie_obj <- recipe({1} ~ ., data=data_set)%>%
                {2}step_dummy(all_predictors(),-all_outcomes())%>%
                step_knnimpute(all_predictors(),-all_outcomes())%>%
                step_center(all_predictors(),-all_outcomes())%>%
                step_scale(all_predictors(),-all_outcomes())%>%
                prep(data=data_set)
            x_train<-bake(recepie_obj, new_data=data_set)
            write.csv(as.matrix(x_train),file = "data/data_ready.csv")
            '''.format(path, target, dummy_bool))

    data_ready = pd.read_csv('data/data_ready.csv')

    output_set = list(set(data_ready[target]))
    global y_trainers
    if reg_val==True:
        data_set=data_ready.values
        y_trainers=data_set[:,1]
    print(list(set(data_ready[target])))
    global num_cat
    num_cat = len(output_set) - 1

    data_ready = np.array(data_ready)


    for i in range(len(data_ready)):
        data_ready[i,1] = output_set.index(data_ready[i,1])

    return data_ready
