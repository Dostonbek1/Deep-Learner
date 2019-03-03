######################################################################
# Revolution UC
# Giorgi, Dostonbek, Hila and Emely

######################################################################
from tkinter import *
import cv2
import tkinter.filedialog as filer
import tkinter as tk
import tkinter
import csv
from PIL import Image, ImageTk


class Datactive:

    def __init__(self):
        self.file = ''
        self.savefile = ''
        self.path = ''
        self.target_model = ''
        self.dropdown = ''
    ###########################################################################
        self.label = tkinter.Label(root, text="Target", bg="Green", fg='white')
        self.label.pack()
        self.label.place(x=20, y=0)
    ############################################################################
        self.target = tkinter.Entry()
        self.target.pack()
        self.target.place(x=70, y=0)

    def buildGUI(self, root):
        self.ImgFrame = Frame(root, width=650, height=500, background="bisque")
        self.ImgFrame.pack()
        self.ImgFrame.place(x=20, y=80)

        choose_file = Button(root, text="Import data", command=self.chooseFile, height=2, width=18, bg='green', fg='white', font=20)
        choose_file.pack()
        choose_file.place(x=700, y=150)

        train = Button(root, text="Train Data", command=self.file, height=1, width=9, bg='green', fg='white', font=20)
        train.pack()
        train.place(x=700, y=300)

        test = Button(root, text="Test Data", command=self.file, height=1, width=9, bg='green', fg='white', font=20)
        test.pack()
        test.place(x=700, y=450)

        target_button = Button(root, text="Submit", command=self.target.get, bg='green', fg='white')
        target_button.pack()
        target_button.place(x=200, y=0)

        # first drop down for Optimizer
        var = tk.StringVar(root)
        # initial value
        var.set('')
        choices = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta','Adam', 'Adamax', 'Nadam']
        option = tk.OptionMenu(root, var, *choices)
        option.pack(side='left', padx=10, pady=10)
        option.place(x=20, y = 30)
        button = tk.Button(root, text="Choose optimizer", command=self.dropdown)
        button.pack(side='left', padx=20, pady=10)
        button.place(x=130, y = 30)

        # second drop down menu /// Class Mode
        var2 = tk.StringVar(root)
        # initial value
        var2.set('')
        choices = ['Binary', 'Multiclass', 'Linear regression']
        option2 = tk.OptionMenu(root, var2, *choices)
        option2.pack(side='left', padx=10, pady=10)
        option2.place(x=300, y = 30)
        button2 = tk.Button(root, text="Choose Class Mode", command=self.dropdown)
        button2.pack(side='left', padx=10, pady=10)
        button2.place(x=370, y = 30)



    def chooseFile(self):
        self.file = filer.askopenfilename(initialdir="/", title="Select file", filetypes=(("excel csv files", "*.csv"), ("all files", "*.*")))
        filename = self.file
        print(filename)
        self.path=filename
        return filename
        # with open(self.file) as file:
        #     file_data = file.read()
        # # print(file_data)

    # we created this function to store the target input but we dont think its working. might not needed but here just in case
    # def retrieve_input(self):
    #     input = self.target.get()
    #     print(input)
    #     return input





def main():
    global test, root
    root = Tk()
    root.geometry('900x700+300+50')
    root.resizable(0, 0)
    root.configure(background='white')
    root.title("Datactive")
    test = Datactive()
    test.buildGUI(root)
    root.mainloop()



if __name__ == '__main__':
    main()
