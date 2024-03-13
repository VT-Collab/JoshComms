#GUI USE
from Tkinter import *

class GUI_Interface(object):
    def __init__(self):
        self.root = Tk()
        self.root.geometry("+100+100")
        self.root.title("Uncertainity Output")
        self.update_time = 0.02
        self.fg = '#ff0000'
        font = "Palatino Linotype"
#"Salt,Pepper,plate,fork,fork2,spoon,spoon2,cup,mug"
        # X_Y Uncertainty
        self.myLabel1 = Label(self.root, text = "Salt", font=(font, 40))
        self.myLabel1.grid(row = 0, column = 0, pady = 30, padx = 30)
        self.textbox1 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox1.grid(row = 1, column = 0,   pady = 50, padx = 100) 
        self.textbox1.insert(0,0)
          
        self.myLabel2 = Label(self.root, text = "Pepper", font=(font, 40))
        self.myLabel2.grid(row = 0, column = 1,pady = 30, padx = 30) 
        self.textbox2 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font,60))
        self.textbox2.grid(row = 1, column = 1,   pady = 50, padx = 100)
        self.textbox2.insert(0,0)
     
        self.myLabel3 = Label(self.root, text = "Plate", font=(font, 40))
        self.myLabel3.grid(row = 0, column = 2, pady = 30, padx = 30) 
        self.textbox3 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox3.grid(row = 1, column = 2,   pady = 50, padx = 100)
        self.textbox3.insert(0,0)

        
        # X_Y Uncertainty
        self.myLabel4 = Label(self.root, text = "Fork Pile", font=(font, 40))
        self.myLabel4.grid(row = 2, column = 0, pady = 30, padx = 30)
        self.textbox4 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox4.grid(row = 3, column = 0,   pady = 50, padx = 100)
        self.textbox4.insert(0,0)

        
        # X_Y Uncertainty
        self.myLabel5 = Label(self.root, text = "Fork Plate", font=(font, 40))
        self.myLabel5.grid(row = 2, column = 1,pady = 30, padx = 30)
        self.textbox5 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox5.grid(row = 3, column = 1, pady = 50, padx = 100)
        self.textbox5.insert(0,0)

        
        # X_Y Uncertainty - 6
        self.myLabel6 = Label(self.root, text = "Spoon Pile", font=(font, 40))
        self.myLabel6.grid(row = 2, column = 2,pady = 30, padx = 30)
        self.textbox6 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox6.grid(row = 3, column = 2,  pady = 50, padx = 100)
        self.textbox6.insert(0,0)
        
                
        # X_Y Uncertainty -- 7
        self.myLabel7 = Label(self.root, text = "Spoon Plate", font=(font, 40))
        self.myLabel7.grid(row = 4, column = 0, pady = 30, padx = 30)
        self.textbox7 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox7.grid(row = 5, column = 0,  pady = 50, padx = 100) 
        self.textbox7.insert(0,0)

        
        # X_Y Uncertainty -- 8
        self.myLabel8 = Label(self.root, text = "Can", font=(font, 40))
        self.myLabel8.grid(row = 4, column = 1, pady = 30, padx = 30) 
        self.textbox8 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox8.grid(row = 5, column = 1,   pady = 50, padx = 100)
        self.textbox8.insert(0,0)

        
        # X_Y Uncertainty --- 9 
        self.myLabel9 = Label(self.root, text = "Mug", font=(font, 40))
        self.myLabel9.grid(row = 4, column = 2,  pady = 30, padx = 30)
        self.textbox9 = Entry(self.root, width = 4, bg = "white", fg=self.fg, borderwidth = 3, font=(font, 60))
        self.textbox9.grid(row = 5, column = 2,  pady = 50, padx = 100)
        self.textbox9.insert(0,0)