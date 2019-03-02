import keras
import copy
from pathlib import *
mnist=keras.datasets.mnist
import pandas as pd
import subprocess

def make_dummy(data):
    s = pd.Series(list('abca'))
    pd.get_dummies(s)

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
