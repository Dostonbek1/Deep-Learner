import keras
import copy
from pathlib import *
import subprocess
import pandas as pd
import rpy2.robjects as robjects
import numpy as np


def make_prep():
    subprocess.call(["R the_recepie_scrip_imputedt.R"], shell=True)

def n_network(data,optimizer_var):
    global num_cat
    multi_class = False
    if num_cat != 1:
        multi_class = True
    # print(x_train[:, 2:])
    if multi_class == False:
        activation_type = 'sigmoid'
        loss_func = 'binary_crossentropy'
        output_layer = 1
        y_train = data[:, 1]
    else:
        activation_type = 'softmax'
        loss_func = 'categorical_crossentropy'
        num_cat += 1
        y_train=keras.utils.to_categorical(data[:, 1],num_classes=num_cat) 
        
    
    x_train = data[:, 2:]
    # print(x_train)
    
    # print(y_train)

    model=keras.models.Sequential()
    model.add(keras.layers.Dense(250,activation="relu"))
    model.add(keras.layers.Dense(250,activation="relu"))

    model.add(keras.layers.Dense(num_cat,activation=activation_type))


    model.compile(optimizer=optimizer_var,
                  loss=loss_func,
                  metrics=["accuracy"],
                  )

    model.fit(x_train,y_train,128,5,validation_split=0.2)
    return model

def save(model):
    model_str=model.to_json()
    f_path=Path("model_str.json")
    f_path.write_text(model_str)
    model.save_weights("model_weights.h5")

def loader(target,path):
# make_prep()

    # dummy_input = input("Does your data set contain non-numeric values? : ")
    dummy_bool = '#'
    # if dummy_input == "Y" or dummy_input == "y":
    #     dummy_bool = ''

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
            write.csv(as.matrix(x_train),file = "data/data_ready.csv")
            '''.format(path, target, dummy_bool))


    data_ready = pd.read_csv('data/data_ready.csv')
    output_set = list(set(data_ready[target]))
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