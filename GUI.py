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
from tkinter import *
import tkinter.filedialog as filer
import tkinter as tk
from PIL import Image, ImageTk
from backend_nn import *
import tkinter
import os


class DeepLearner:
    """
    This class initializes all the variables used in the interface and in the backend"
    """
    def __init__(self):
        self.file = ''
        self.savefile = ''
        self.path = ''
        self.dropdown = ''
        self.targeter=''
        self.layer_reg=[]
        self.hidden_col = 5
        self.layer_nodes_lst = []
        self.density_matrix = []
        self.drop_or_not=[]
        self.dummy_check_value = tkinter.BooleanVar()
        self.regression_status = tkinter.BooleanVar()
        self.dummy_check_status = False
        self.regur_status=False
        self.username=None
        self.password=None
        self.col=1
    def buildGUI(self, root):
        """

        :param root: The root of the tkinter
        :return: None
        """
        self.root=root
        im = Image.open("images/logo.png")
        photo = ImageTk.PhotoImage(im)

        label = Label(root, image=photo)
        label.image = photo  # keep a reference!
        label.pack()
        label.place(x=600, y=5)

        self.ModelVizFrame = Frame(root, width=950, height=500, background="bisque")
        self.ModelVizFrame.pack()
        self.ModelVizFrame.place(x=40, y=160)

        choose_file = Button(root, text="Import data", command=self.chooseFile, height=2, width=18, bg='green', fg='white', font=20)
        choose_file.pack()
        choose_file.place(x=1180, y=125)

        viz_data = Button(root, text="Visualize Data w/ Tableau", command=self.viz_data, height=2, width=20, bg='blue', fg='white', font=20)
        viz_data.pack()
        viz_data.place(x=1170, y=175)

        train = Button(root, text="Train Data", command=self.train_data, height=2, width=18, bg='green', fg='white', font=20)
        train.pack()
        train.place(x=1180, y=300)

        save_model = Button(root, text="Save Model", command=self.save_model, height=2, width=18, bg='green', fg='white', font=20)
        save_model.pack()
        save_model.place(x=1180, y=350)


        reset_btn = Button(root, text="Reset", command=self.reset, height=2, width=18, bg='green', fg='white', font=20)
        reset_btn.pack()
        reset_btn.place(x=1180, y=475)

        self.target_box = tkinter.Entry()
        self.target_box.insert(0, "Target...")
        self.target_box.pack()
        self.target_box.place(x=50, y=125)

        self.epoch_box = tkinter.Entry()
        self.epoch_box.insert(0, "Epochs")
        self.epoch_box.pack()
        self.epoch_box.place(x=220, y=125)

        self.batch_box = tkinter.Entry()
        self.batch_box.insert(0, "Batch size")
        self.batch_box.pack()
        self.batch_box.place(x=390, y=125)

        self.validation_box = tkinter.Entry()
        self.validation_box.insert(0, "Validation Split")
        self.validation_box.pack()
        self.validation_box.place(x=390, y=100)

        self.dummy_check_box = tkinter.Checkbutton(root, text="Contains Qualitative", variable=self.dummy_check_value)
        self.dummy_check_box.pack()
        self.dummy_check_box.place(x=560, y=125)

        # /// Class Mode check box
        self.regression_checkbox = tkinter.Checkbutton(root, text="Regression", variable=self.regression_status)
        self.regression_checkbox.pack()
        self.regression_checkbox.place(x=730, y=125)

        # first drop down for Optimizer
        self.var = tk.StringVar(root)
        # initial value
        self.var.set('Choose Optimizer')
        choices = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta','Adam', 'Adamax', 'Nadam']
        self.option = tk.OptionMenu(root, self.var, *choices)
        self.option.pack(side='left', padx=10, pady=10)
        self.option.place(x=860, y = 120)

        # initializing layers
        self.input_layer = self.create_layer("Input", 0, 0)
        self.add_layer = Button(self.ModelVizFrame, text="+", command=self.create_new_layer, height=1, width=1, bg='green', fg='white')
        self.add_layer.grid(row=0, column=99)

        self.output_layer = self.create_layer("Output", 0, 100, "Output")

        
    def create_layer(self, layer_label, layer_row, layer_col, layer="Not Output"):
        """
        Creates the layer on the GUI display.
        :param layer_label: Label of the layer
        :param layer_row: row position of the layer
        :param layer_col: column position of the layer
        :param layer: Layer type
        :return:
        """
        layer_label = tk.Label(self.ModelVizFrame, text=layer_label, width=5, bg='lightblue', anchor='w', pady=4, font=('Verdana', 8, 'bold'))
        layer_label.grid(row=layer_row, column=layer_col, sticky='e')

        im = Image.open("images/layer.png")
        photo = ImageTk.PhotoImage(im)

        label = Label(self.ModelVizFrame, image=photo)
        label.image = photo  # keep a reference!
        label.grid(row=layer_row+1, column=layer_col)


        if layer != "Output":

            im2 = Image.open("images/weights.png")
            photo2 = ImageTk.PhotoImage(im2)

            label2 = Label(self.ModelVizFrame, image=photo2)
            label2.image = photo2  # keep a reference!
            label2.grid(row=layer_row+1, column=layer_col+1)

            target_box = tkinter.Entry(self.ModelVizFrame, width=4)
            target_box.insert(0, "10")
            target_box.grid(row=layer_row+2, column=layer_col)
            
            self.layer_nodes_lst.append(target_box)

    
    def create_new_layer(self):
        """
        Creates a layer on the GUI
        :return: None
        """
        if self.hidden_col < 17:
            layer_type = tk.StringVar(self.ModelVizFrame)
            layer_type.set('Layer')
            choices = ['Dense', 'Drop Out']
            layer_option = tk.OptionMenu(self.ModelVizFrame, layer_type, *choices)
            layer_option.grid(row=0, column = self.hidden_col)


            im = Image.open("images/layer.png")
            photo = ImageTk.PhotoImage(im)

            label = Label(self.ModelVizFrame, image=photo)
            label.image = photo  # keep a reference!
            label.grid(row=1, column=self.hidden_col)

            im2 = Image.open("images/weights.png")
            photo2 = ImageTk.PhotoImage(im2)

            label2 = Label(self.ModelVizFrame, image=photo2)
            label2.image = photo2  # keep a reference!
            label2.grid(row=1, column=self.hidden_col+1)

            target_box = tkinter.Entry(self.ModelVizFrame, width=4)
            target_box.insert(0, "10")
            target_box.grid(row=2, column=self.hidden_col)
            
            self.layer_nodes_lst.append(target_box)
            self.drop_or_not.append(layer_type)
            self.hidden_col += 1

        # function to train data by getting the user selections
    def train_data(self):
        """
        The function to train the data and complete the whole thing
        :return: None
        """
        self.density_matrix = []
        self.data_ready = []
        self.optimizer_value = self.var.get()
        self.density_matrix.append([int(self.layer_nodes_lst[0].get()),"Dense"])

        for layer_num in range(len(self.layer_nodes_lst)-1):
            self.density_matrix.append([int(self.layer_nodes_lst[layer_num+1].get()),self.drop_or_not[layer_num].get()])
            print(self.density_matrix)

        self.targeter = self.target_box.get()
        self.epochs_size = int(self.epoch_box.get())
        self.batch_size = int(self.batch_box.get())
        self.validation_split=int(self.validation_box.get())/100
        self.data_ready = loader(self.targeter, self.path, self.dummy_check_value.get(),self.regression_status.get())
        self.modeler=Network()
        self.model, self.hist = self.modeler.n_network(self.data_ready, self.optimizer_value, self.density_matrix, self.batch_size, self.epochs_size, self.regression_status.get(),self.validation_split)
        self.modeler.ploter(self.hist)


    def viz_data(self,file_def='C:/Program Files/Tableau/Tableau 2019.1/bin/tableau.exe'):
        """
        Visualizing the data with Tableau
        :param file_def: Tableau file location by default
        :return:
        """
        try:
            os.startfile(file_def)
        except:
            m = Tk()
            m.wm_title("Select Tableau")
            label = Label(m, text="In order to use this feature please install tableau.exe on your computer, and place it in C:/Program Files/")
            label.pack(side="top", fill="x", pady=10)
            B1 = Button(m, text="Okay", command=m.destroy)
            B1.pack()

        
    def dummy_handler(self):
        """
        The call to the dummy handler
        :return:  None
        """
        print(self.dummy_check_value.get())
        if self.dummy_check_value.get() == "off":
            self.dummy_check_status = False
        else:
            self.dummy_check_status = True

         
    def chooseFile(self):
        """
        Choose the data.csv file to train on
        :return: None
        """
        self.file = filer.askopenfilename(initialdir="/", title="Select file", filetypes=(("excel csv files", "*.csv"), ("all files", "*.*")))
        filename = self.file
        print(filename)
        self.path=filename
    

    def reset(self):
        """
        Restart the GUI
        :return:
        """
        global root
        root.destroy()
        main()

    def save_model(self):
        self.modeler.save(self.model)

def main():
    """
    calling all the functions 
    """
    global test, root
    root = Tk()
    root.geometry('1200x700+300+50')
    root.configure(background='white')
    root.title("Datactive")
    test = DeepLearner()
    test.buildGUI(root)
    root.mainloop()



if __name__ == '__main__':
    main()
