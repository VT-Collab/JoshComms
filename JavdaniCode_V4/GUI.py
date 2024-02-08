#GUI USE
from tkinter import *

class GUI_Interface(object):
    def __init__(self):
        self.root = Tk()
        self.root.geometry("+100+100")
        self.root.title("Uncertainity Output")
        self.update_time = 0.02
        self.fg = '#ff0000'
        font = "Palatino Linotype"

        # X_Y Uncertainty
        self.myLabel1 = Label(self.root, text = "Red", font=(font, 80))
        self.myLabel1.grid(row = 0, column = 0, pady = 50, padx = 50)
        self.textbox1 = Entry(self.root, width = 8, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 80))
        self.textbox1.grid(row = 1, column = 0,  pady = 50, padx = 100)
        self.textbox1.insert(0,0)
          
        self.myLabel2 = Label(self.root, text = "Yellow", font=(font, 80))
        self.myLabel2.grid(row = 0, column = 1, pady = 50, padx = 50)
        self.textbox2 = Entry(self.root, width = 8, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 80))
        self.textbox2.grid(row = 1, column = 1,  pady = 50, padx = 100)
        self.textbox2.insert(0,0)
     
        self.myLabel3 = Label(self.root, text = "Green", font=(font, 80))
        self.myLabel3.grid(row = 0, column = 2, pady = 50, padx = 50)
        self.textbox3 = Entry(self.root, width = 8, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 80))
        self.textbox3.grid(row = 1, column = 2,  pady = 50, padx = 100)
        self.textbox3.insert(0,0)