import keras
import copy
from pathlib import *
import subprocess
import pandas as pd
import rpy2.robjects as robjects
import numpy as np
from ann_visualizer.visualize import ann_viz
from keras.callbacks import History
import matplotlib.pyplot as plt
import tkinter as tk
# from test import test


def make_prep():
    subprocess.call(["R the_recepie_scrip_imputedt.R"], shell=True)
def tester(model,data):
    global num_cat
    multi_class = False
    # print(x_train[:, 2:])
    if multi_class == False and not regression_status:
        y_train = data[:, 1]
    
    else:
        y_train=keras.utils.to_categorical(data[:, 1],num_classes=num_cat) 
        
    if regression_status:
        y_train = y_trainers
        x_train = data[:, 2:]
     
    model.evaluate(x_train,y_train,100,verbose=1)

def n_network(data,optimizer_var, layers, batch_sz, epochs_size, regression_status):
    global num_cat
    multi_class = False
    metrics_value = 'accuracy'
    if num_cat != 1:
        multi_class = True
    # print(x_train[:, 2:])
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
    # print(x_train)
    
    # print(y_train)

    model=keras.models.Sequential()
    # model.add(keras.layers.Dense(250,activation="relu"))
    if len(layers) > 0:
        for layer in layers:
            model.add(keras.layers.Dense(layer[0],activation="relu"))
    # model.add(keras.layers.Dense(250,activation="relu"))
    if regression_status:
        model.add(keras.layers.Dense(1))
    else:
        model.add(keras.layers.Dense(num_cat,activation=activation_type))


    model.compile(optimizer=optimizer_var,
                  loss=loss_func,
                  metrics=[metrics_value],
                  )
    hist=History()
    model.fit(x_train,y_train,batch_sz,epochs_size,validation_split=0.2,callbacks=[hist])
    # messagebox_status.destroy()
    messagebox = tk.messagebox.showinfo("Info", "Done Training.")
    ploter(hist)
    # ploter(hist["acc"],hist["val_acc"],hist["loss"],hist[val_loss])
    # ann_viz(model, title="My first neural network")
    return model

def ploter(hist):
    hist_keys=list(hist.history.keys())
    print("hist keys",hist_keys)
    plt.plot(hist.history[hist_keys[1]],'ro-', label=hist_keys[1])
    plt.plot(hist.history[hist_keys[3]],'go-', label=hist_keys[3])
    plt.title('Model Accuracy')
    plt.ylabel(hist_keys[3])
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def save(model):
    model_str=model.to_json()
    f_path=Path("output/model_str.json")
    f_path.write_text(model_str)
    model.save_weights("output/model_weights.h5")

def loader(target,path,dummy_input,reg_val,test_bool=False):
# make_prep()
    tets_switch=""
    if test_bool:
        tets_switch="_test"
    # dummy_input = input("Does your data set contain non-numeric values? : ")
    dummy_bool = '#'
    if dummy_input:
        dummy_bool = ''

    robjects.r(r'''
            library(recipes)
            data_set <- read.csv("{0}")
            data_set <- data_set %>% select({1}, everything())
            recepie_obj <- recipe({1} ~ ., data=data_set)%>%
                {2}step_dummy(all_predictors(),-all_outcomes())%>%
                step_knnimpute(all_predictors(),-all_outcomes())%>%
                step_center(all_predictors(),-all_outcomes())%>%
                step_scale(all_predictors(),-all_outcomes())%>%
                prep(data=data_set)
            x_train<-bake(recepie_obj, new_data=data_set)
            write.csv(as.matrix(x_train),file = "data/data_ready{3}.csv")
            '''.format(path, target, dummy_bool,tets_switch))

    if test_bool:
        data_ready = pd.read_csv('data/data_ready_test.csv')
    else:
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



# print(data_ready)

# file_path = input("What is the file path of your data: ")
# df = pd.read_csv(file_path)
# target_outcome = input("What is your target?:")

# data_ready = loader(target_outcome,file_path)

# # print(data_ready)


# n_network(data_ready)