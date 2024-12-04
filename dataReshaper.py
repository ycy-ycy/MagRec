#######################
##                   ##
## package importing ##
##                   ##
#######################

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

######################
##                  ##
## global variables ##
##                  ##
######################

bX = np.zeros((1,1))
bY = np.zeros((1,1))
bZ = np.zeros((1,1))
b3D = np.zeros((1,1,3))
bLoaded = False

##############################
##                          ##
## main application windows ##
##                          ##
##############################

app = tk.Tk()

def on_closing():
    app.destroy()
    app.quit()

window_width = 240
window_height = 80
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

app.title("data reshaper")
app.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
app.attributes("-topmost", True)
app.attributes("-topmost", False)
app.protocol("WM_DELETE_WINDOW", on_closing)

###################
##               ##
## function pool ##
##               ##
###################

def loadB3():
    global b3D, bLoaded, bX, bY, bZ

    loadFlag=False
    shapeFlag=False
    
    bXPath = filedialog.askopenfilename(title="Select Bx Data (in .npy format)")
    try:
        bXT = np.load(bXPath)
        if len(bXT.shape)!=2:
            shapeFlag=True
    except:
        loadFlag=True

    bYPath = filedialog.askopenfilename(title="Select By Data (in .npy format)")
    try:
        bYT = np.load(bYPath)
        if bYT.shape != bXT.shape:
            shapeFlag=True
    except:
        loadFlag=True
    
    bZPath = filedialog.askopenfilename(title="Select Bz Data (in .npy format)")
    try:
        bZT = np.load(bZPath)
        if bZT.shape != bXT.shape:
            shapeFlag=True
    except:
        loadFlag=True

    if loadFlag:
        messagebox.showinfo("Loading Error", "Failed to load selected file or file selection canceled")
    elif shapeFlag:
        messagebox.showinfo("Shape Error", "B data must be 2D arrays with the same size")
    else:
        bX = bXT
        bY = bYT
        bZ = bZT
        b3D = np.concatenate((bX[:,:,np.newaxis],bY[:,:,np.newaxis],bZ[:,:,np.newaxis]),axis=2)
        bLoaded = True

    return

def loadB1():
    global b3D, bLoaded, bX, bY, bZ
    
    bPath = filedialog.askopenfilename(title="Select B Data (in .npy format)")
    try:
        bT = np.load(bPath)
        if len(bT.shape)!=3:
            messagebox.showinfo("Shape Error", "B data must be 3D array")
            return
    except:
        messagebox.showinfo("Loading Error", "Failed to load selected file or file selection canceled")
        return
    
    b3D = bT
    bLoaded = True
    bX = b3D[:,:,0]
    bY = b3D[:,:,1]
    bZ = b3D[:,:,2]

    return

def saveB3():
    if not bLoaded:
        messagebox.showinfo("Data Error", "No data loaded")
        return

    bXpath = filedialog.asksaveasfilename(title="Save Bx Data (in .npy format)", defaultextension=".npy", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    bYpath = filedialog.asksaveasfilename(title="Save By Data (in .npy format)", defaultextension=".npy", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    bZpath = filedialog.asksaveasfilename(title="Save Bz Data (in .npy format)", defaultextension=".npy", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if bXpath == "" or bYpath == "" or bZpath == "":
        messagebox.showinfo("Save Error", "Save canceled")
        return
    
    np.save(bXpath, bX)
    np.save(bYpath, bY)
    np.save(bZpath, bZ)

    return

def saveB1():
    if not bLoaded:
        messagebox.showinfo("Data Error", "No data loaded")
        return

    bPath = filedialog.asksaveasfilename(title="Save B Data (in .npy format)", defaultextension=".npy", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if bPath == "":
        messagebox.showinfo("Save Error", "Save canceled")
        return
    
    np.save(bPath, b3D)

    return

##############################
##                          ##
## components on the window ##
##                          ##
##############################

frame = tk.Frame(app)
frame.pack(anchor="center", fill="both", expand=True)

loadB1Button = tk.Button(frame, text="Load 1 File", command=loadB1, width=15)
saveB1Button = tk.Button(frame, text="Save to 1 File", command=saveB1, width=15)
loadB3Button = tk.Button(frame, text="Load 3 Files", command=loadB3, width=15)
saveB3Button = tk.Button(frame, text="Save to 3 Files", command=saveB3, width=15)

loadB1Button.grid(row=0, column=0, padx=3, pady=5)
saveB3Button.grid(row=0, column=1, padx=3, pady=5)
loadB3Button.grid(row=1, column=0, padx=3, pady=5)
saveB1Button.grid(row=1, column=1, padx=3, pady=5)

app.mainloop()