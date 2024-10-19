import tkinter as tk


"""
this file will contain another class for another GUI which has the sole purpose of defining new models and training them, this GUI will have toplevels which display
graphs of the various infomation of the training process aswell as displaying the values of the losses and the mAP during training
"""

class ModelTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Trainer")