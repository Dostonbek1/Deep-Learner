######################################################################
# Revolution UC
# Giorgi, Dostonbek, Hila and Emely
# RevolutionUC
# Project: Deep Learner
#
# Authors:
#
# •	Giorgi Lomia •	Dostonbek Toirov •	Hila Manalai •	Emely Alfaro
#
# Categories: •	Education •	Demystify Data
#
# Purpose: To make neural networks/deep learning more accessible to general public and those who might not have programming skills.
#
# Description: Deep Learner is an interactive GUI that allows users to import data, visualize it, decide which variable to predict and run neural networks on it. Users are able to construct and train complex neural networks without writing any code. Based on the obtained results, users can try more optimizers, change predictors or save and export the model results to make decisions. In general, this program enables users to create their own neural network models without any prior knowledge of coding.
#
# What we learned: •	Creating a fully viable product •	Better understanding of Deep Learning and the underlying structure of neural networks •	Designing user interface •	Optimizing user experience •	Understanding user needs
#
# To Run the App
# Clone or download the repo as .zip file
# You need to have Python 3 to be able to run the app. Install requirements and dependencies by running:
# $ pip3 install -r requirements.txt
# Start the app by running:
# $ python3 GUI.py
######################################################################

import keras
import time
from pathlib import *
import pandas as pd
import numpy as np
from keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler


class Network:
    """
    The network that is created in the background from the GUI inputs.
    """
    def __init__(self):
        """
        The model class.
        """
        self.model = keras.models.Sequential()

    def n_network(self, data, optimizer_var, layers, batch_sz, epochs_size, regression_status, validation_spliter):
        """
        Builds the network depending on the user input.
        :param data: Data to train on
        :param optimizer_var: Optimizer chosen
        :param layers: List of layers and their values
        :param batch_sz: The batch size
        :param epochs_size: Number of epochs
        :param regression_status: True or False regression or not
        :param validation_spliter: The validation split
        :return: The model and the history of the fit
        """
        global num_cat
        multi_class = False
        metrics_value = 'accuracy'
        if num_cat != 1:
            multi_class = True
        if not multi_class and not regression_status:
            activation_type = 'sigmoid'
            loss_func = 'binary_crossentropy'
            y_train = data[:, 0]

        else:
            activation_type = 'softmax'
            loss_func = 'categorical_crossentropy'
            num_cat += 1
            y_train = keras.utils.to_categorical(data[:, 0],num_classes=num_cat)

        if regression_status:
            loss_func = 'mean_squared_error'
            y_train = y_trainers
            metrics_value = 'mae'

        # print(type(y_train))
        x_train = data[:, 1:]
        print(x_train)
        print(y_train)

        if len(layers) > 0:
            for layer in layers:
                print(layer)
                if layer[1] == "Drop Out":
                    self.add_dropout(layer[0]/100)
                else:
                    self.add_dense_layer(layer[0], "relu")
        if regression_status:
            self.model.add(keras.layers.Dense(1))
        else:
            self.model.add(keras.layers.Dense(num_cat, activation=activation_type))

        self.model.compile(optimizer=optimizer_var, loss=loss_func, metrics=[metrics_value])
        hist=History()
        self.model.fit(x_train, y_train, batch_sz, epochs_size, validation_split=validation_spliter, callbacks=[hist], verbose=1)

        print(self.model.summary())
        return self.model, hist

    def add_dense_layer(self, neurons, act):
        """
        Add a dense layer to the network class.
        :param neurons: Number of neurons
        :param act: activation function
        :return: None
        """
        self.model.add(keras.layers.Dense(neurons, activation=act))

    def add_dropout(self, rater):
        """
        Adding a dropout layer.
        :param rater: The rate of drop out
        :return: None
        """
        self.model.add(keras.layers.Dropout(rate=rater))

    def ploter(self, hist):
        """
        Plots the model metrics.
        :param hist: The model history to plot
        :return: None
        """
        hist_keys = list(hist.history.keys())
        print("hist keys", hist_keys)
        plt.plot(hist.history[hist_keys[1]], 'ro-', label=hist_keys[1])
        plt.plot(hist.history[hist_keys[3]], 'go-', label=hist_keys[3])
        plt.title('Model Accuracy')
        plt.ylabel(hist_keys[3])
        plt.xlabel('epoch')
        plt.legend()
        plt.show(block=True)
        time.sleep(5)
        plt.close("all")

    def save(self, model):
        """
        Saves the model as a json file.
        :param model: The model
        :return: None
        """
        model_str = model.to_json()
        f_path = Path("output/model_str.json")
        model.save('output/model_str.h5')
        model.save_weights("output/model_weights.h5")


def find_factors(df):
    """
    This function finds all the possible factor variables.
    :param df: The data frame
    :return: the list of all factor variable names
    """
    fact = []
    for i in range(len(df.columns)):
        print(type(df.iloc[0, i]))
        if type(df.iloc[0, i]) == str:

            fact.append(df.columns[i])

    print(fact)
    if len(fact) != 1:
        return fact[1:]
    return fact


def preprocess(df, target, dummy):
    """
    This Prepossesses the data frame there is automation involved.
    :param df: The data Frame
    :param target: The target we are looking for
    :param dummy: True or False if the data contains factor values
    :return: Preprocessed DataFrame
    """
    cols = list(df.columns.values)
    cols.pop(cols.index(target))
    df = df[[target]+cols]
    targ = df[target]

    factors = find_factors(df)
    if dummy:
        df = pd.get_dummies(df.iloc[:, 1:], columns=factors,prefix=list(factors)).rename(columns=lambda x: 'Category_' + str(x))
        print(df.columns)

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(df.iloc[:, 1:])
    df = pd.DataFrame(imp.transform(df.iloc[:, 1:]))

    # Scaling
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]))
    df[target] = targ
    cols = list(df.columns.values)
    cols.pop(cols.index(target))
    df = df[[target] + cols]
    return df


def loader(target, path, dummy_input, reg_val):
    """
    This function load the data from the path provided by the user and and prepossesses it and structures it properly
    and that gets passed into the network.
    :param target: The target we are trying to predict or classify
    :param path: The path to the data
    :param dummy_input: True or False if the data contains factor values
    :param reg_val: True or False if the model needs to be regression
    :return: DataFrame that is ready to train on
    """
    # Read the csv file from the path provided
    df = pd.read_csv(path)

    # Pass the data into the prepossess function
    data_ready = preprocess(df, target, dummy_input)

    # This generates unique values in the target
    output_set = list(set(data_ready[target]))

    global y_trainers
    # If the problem is a regression problem
    if reg_val:
        data_set = data_ready.values
        y_trainers = data_set[:, 0]

    # Otherwise generate the number of categories
    global num_cat
    num_cat = len(output_set) - 1
    data_ready = np.array(data_ready)

    # Replaces the the categories with their numerical values
    for i in range(len(data_ready)):
        data_ready[i, 0] = output_set.index(data_ready[i, 0])

    return data_ready
