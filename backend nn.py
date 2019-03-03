import keras
import copy
from pathlib import *
import subprocess
import pandas as pd
import rpy2.robjects as robjects



def make_prep():
    subprocess.call(["R the_recepie_scrip_imputedt.R"], shell=True)

def n_network(data):

    x_train=keras.utils.normalize(data,1)
    y_train = keras.utils.to_categorical(data[0], 10)


    model=keras.models.Sequential()
    model.add(keras.layers.Dense(250,activation="relu",input_shape=len(x_train)))
    model.add(keras.layers.Dense(250,activation="relu"))
    model.add(keras.layers.Dense(10,keras.activations.softmax))


    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"],
                  )

    model.fit(x_train,y_train,128,5,validation_split=0.2)
    return model

def save(model):
    model_str=model.to_json()
    f_path=Path("model_str.json")
    f_path.write_text(model_str)
    model.save_weights("model_weights.h5")

df=pd.read_csv("data/HR_Churn.csv")
# make_prep()

target_outcome = input("What is your target?:")

robjects.r(r'''
        library(recipes)
        data_set <- read.csv("data/HR_Churn.csv")

        recepie_obj <- recipe({0} ~ ., data=data_set)%>%
            step_dummy(all_predictors(),-all_outcomes())%>%
            step_knnimpute(all_predictors(),-all_outcomes())%>%
            step_center(all_predictors(),-all_outcomes())%>%
            step_scale(all_predictors(),-all_outcomes())%>%
            prep(data=data_set)
        x_train<-bake(recepie_obj, new_data=data_set)
        write.csv(as.matrix(x_train),file = "data/data_ready.csv")
        '''.format(target_outcome))



        