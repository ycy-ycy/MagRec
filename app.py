# Magnetization Reconstruction
# Copyright (C) 2024  Changyu Yao

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#######################
##                   ##
## package importing ##
##                   ##
#######################

import cupy as cp
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from numba import jit, prange
import cv2
import gc
from PIL import Image
import re
from datetime import datetime
import psutil
import webbrowser

##########################
##                      ##
## all global variables ##
##                      ##
##########################

width, height, distance, xShift, yShift = np.float64(144), np.float64(144), np.float64(8), np.float64(0), np.float64(0)

bExp = cp.zeros(shape=(1,1,1),dtype=cp.float64)
bPixelX, bPixelY = 1,1
bLoaded = False

bStartX, bStartY, bEndX, bEndY, bCropX, bCropY = 0,0,bPixelX,bPixelY,bPixelX,bPixelY
bRMS = np.sqrt(cp.sum(bExp**2)/bPixelX/bPixelY/3)
bCropped = cp.copy(bExp)
bCroppedRMS = np.sqrt(cp.sum(bCropped**2)/bCropX/bCropY/3)

mPixelX, mPixelY, NofP, listX, listY, mRes = 1,1,1,np.array([]),np.array([]),np.array([])
mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
weight = np.array([])
regionLoaded = False

matrixA = cp.zeros(shape=(bCropX,bCropY,NofP,3,3),dtype=cp.float64)
AGenerated = False

gLoss = 1.

fromZero = True
pars = "2e-5,2e-5,2e-3,-1,2000,0.005,20"
rate, dist = 1, 0.02

fitD = True
fitX = False
fitY = False
fitRate, fitDist = 0.9, 0.05
fitRound = 8

def hideWindow(f):
    def wrapper(*args, **kwargs):
        app.withdraw()
        result = f(*args, **kwargs)
        app.deiconify()
        return result
    return wrapper


##############################
##                          ##
## main application windows ##
##                          ##
##############################

app = tk.Tk()

def on_closing():
    response = messagebox.askyesno("Close Experiment", "Do you want to close the experiment?")
    if response:  # If the user clicks "Yes"
        app.destroy()
        app.quit()
    else:  # If the user clicks "No"
        return

window_width = 840
window_height = 550
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

app.title("magnetization reconstruct app")
app.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
app.attributes("-topmost", True)
app.attributes("-topmost", False)
app.protocol("WM_DELETE_WINDOW", on_closing)

copyright_label = tk.Label(app, text="Licensed under the GNU General Public License v3.0", justify='right', anchor='e')
copyright_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

########################
##                    ##
## set physical scale ##
##                    ##
########################

scaleFrame = tk.Frame(app)
scaleFrame.pack(pady=(20,10), anchor='center')

scaleLabel = tk.Label(scaleFrame, text="Physical Scale", font=("Helvetica", 16))
scaleLabel.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=5)

scaleBlank = tk.Label(scaleFrame,text="",width=0)
scaleBlank.grid(row=0,column=1,padx=1,pady=5)

scaleFrame1 = tk.Frame(scaleFrame)
scaleFrame1.grid(row=0, column=2, padx=(10,10))

widthLabel = tk.Label(scaleFrame1, text="width=%s"%(width), width=14)
widthLabel.grid(row=0, column=0, padx=(5, 2))
widthEntry = tk.Entry(scaleFrame1, width=14)
widthEntry.grid(row=0, column=1, padx=(2, 10))

heightLabel = tk.Label(scaleFrame1, text="height=%s"%(height), width=14)
heightLabel.grid(row=0, column=2, padx=(5, 2))
heightEntry = tk.Entry(scaleFrame1, width=14)
heightEntry.grid(row=0, column=3, padx=(2, 10))

distanceLabel = tk.Label(scaleFrame1, text="distance=%s"%(distance), width=14)
distanceLabel.grid(row=0, column=4, padx=(5, 2))
distanceEntry = tk.Entry(scaleFrame1, width=14)
distanceEntry.grid(row=0, column=5, padx=(2, 10))

scaleFrame2 = tk.Frame(scaleFrame)
scaleFrame2.grid(row=1, column=2, padx=(10,10))

xShiftLabel = tk.Label(scaleFrame2, text="xShift=%s"%(xShift), width=14)
xShiftLabel.grid(row=0, column=0, padx=(5, 2))
xShiftEntry = tk.Entry(scaleFrame2, width=14)
xShiftEntry.grid(row=0, column=1, padx=(2, 8))

yShiftLabel = tk.Label(scaleFrame2, text="yShift=%s"%(yShift), width=14)
yShiftLabel.grid(row=0, column=2, padx=(5, 2))
yShiftEntry = tk.Entry(scaleFrame2, width=14)
yShiftEntry.grid(row=0, column=3, padx=(4,15))

def setScale():
    global width, height, distance, xShift, yShift, AGenerated, mAVG

    try:
        width = np.float64(widthEntry.get())
    except:
        pass
    try:
        height = np.float64(heightEntry.get())
    except:
        pass
    try:
        distance = np.float64(distanceEntry.get())
    except:
        pass
    try:
        xShift = np.float64(xShiftEntry.get())
    except:
        pass
    try:
        yShift = np.float64(yShiftEntry.get())
    except:
        pass

    widthLabel.config(text="width=%s"%(str(width) if len(str(width))<=10 else "{:.4e}".format(width)))
    heightLabel.config(text="height=%s"%(str(height) if len(str(height))<=9 else "{:.3e}".format(height)))
    distanceLabel.config(text="distance=%s"%(str(distance) if len(str(distance))<=7 else "{:.1e}".format(distance)))
    xShiftLabel.config(text="xShift=%s"%(str(xShift) if len(str(xShift))<=9 else "{:.3e}".format(xShift)))
    yShiftLabel.config(text="yShift=%s"%(str(yShift) if len(str(yShift))<=9 else "{:.3e}".format(yShift)))
    mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
    AGenerated = False
    ACheckVar.set(0)

    return

scaleButton = tk.Button(scaleFrame2, text="set scale", width=9, command=setScale)
scaleButton.grid(row=0, column=4, padx=(9,9))

unitLabel = tk.Label(scaleFrame2, text="arbitrary unit", width=14)
unitLabel.grid(row=0, column=5, padx=(8, 4))

#######################
##                   ##
## load B field data ##
##                   ##
#######################

loadBFrame = tk.Frame(app)
loadBFrame.pack(pady=(10,10), anchor='center')

loadBLabel = tk.Label(loadBFrame, text="Load & Crop\nB Field Data", font=("Helvetica", 16))
loadBLabel.grid(row=0, column=0, rowspan=2, padx=(8, 28), pady=5)

loadBFrame1 = tk.Frame(loadBFrame)
loadBFrame1.grid(row=0, column=2, padx=10)

def loadB():
    global bExp, bPixelX, bPixelY, bLoaded, bCropX, bCropY, bCropped, bRMS, bStartX, bStartY, bEndX, bEndY, mAVG, bCroppedRMS, AGenerated

    loadFlag=False
    shapeFlag=False
    
    bXPath = filedialog.askopenfilename(title="Select Bx Data (in .npy format)")
    try:
        bX = np.load(bXPath)
        if len(bX.shape)!=2:
            shapeFlag=True
    except:
        loadFlag=True

    bYPath = filedialog.askopenfilename(title="Select By Data (in .npy format)")
    try:
        bY = np.load(bYPath)
        if bY.shape != bX.shape:
            shapeFlag=True
    except:
        loadFlag=True
    
    bZPath = filedialog.askopenfilename(title="Select Bz Data (in .npy format)")
    try:
        bZ = np.load(bZPath)
        if bZ.shape != bX.shape:
            shapeFlag=True
    except:
        loadFlag=True

    if loadFlag:
        messagebox.showinfo("Loading Error", "Failed to load selected file or file selection canceled")
    elif shapeFlag:
        messagebox.showinfo("Shape Error", "B data must be 2D arrays with the same size")
    else:
        bExp = cp.asarray(np.concatenate((bX[:,:,np.newaxis],bY[:,:,np.newaxis],bZ[:,:,np.newaxis]),axis=2))
        bCropped = cp.copy(bExp)
        bStartX, bStartY = 0,0
        bEndX, bEndY = bX.shape
        bPixelX, bPixelY = bX.shape
        bCropX, bCropY = bX.shape
        bPixelXLabel.config(text="B width pixel=%s"%(bPixelX))
        bPixelYLabel.config(text="B height pixel=%s"%(bPixelY))
        bLoaded = True
        bCheckVar.set(int(bLoaded))
        bRMS = float(cp.sqrt(cp.sum(bExp**2)/bPixelX/bPixelY/3))
        bRMSLabel.config(text="B RMS=%s"%("{:.6e}".format(float(bRMS))))
        bCroppedRMS = bRMS
        croppedRMSLabel.config(text="ROI RMS=%s"%("{:.4e}".format(float(bCroppedRMS))))
        bCropXLabel.config(text="ROI width=%s"%(bCropX))
        bCropYLabel.config(text="ROI height=%s"%(bCropY))
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        ASizeLabel.config(text="size(est.): %.2fGB"%(72*bCropX*bCropY*NofP/1024/1024/1024))
        AGenerated = False
        ACheckVar.set(0)

    return

loadBButton = tk.Button(loadBFrame1, text="load B data", command=loadB)
loadBButton.grid(row=0, column=0, padx=(5, 5))

bStatus = tk.Label(loadBFrame1, text="B data loaded:", width=14)
bStatus.grid(row=0, column=1, padx=(5, 0))

bCheckVar = tk.IntVar(value=int(bLoaded))
bCheckBox = tk.Checkbutton(loadBFrame1, variable=bCheckVar, state='disabled')
bCheckBox.grid(row=0, column=1, padx=(112,0))

bPixelXLabel = tk.Label(loadBFrame1, text="B width pixel=%s"%(bPixelX), width=14)
bPixelXLabel.grid(row=0, column=2, padx=(5, 5))

bPixelYLabel = tk.Label(loadBFrame1, text="B height pixel=%s"%(bPixelY), width=14)
bPixelYLabel.grid(row=0, column=3, padx=(5, 5))

bRMSLabel = tk.Label(loadBFrame1, text="B RMS=%s"%(0.0), width=16)
bRMSLabel.grid(row=0,column=4,padx=5)

loadBFrame2 = tk.Frame(loadBFrame)
loadBFrame2.grid(row=1, column=2, padx=10)

def makeXTicks(N):
    Pw = int(np.log10(N))
    firstOne = int(str(N)[0])
    firstTwo = int(str(N)[:2])
    if firstOne >= 5:
        ticks = [i*10**Pw for i in range(firstOne+1)]
    elif firstTwo <= 20:
        ticks = [i*10**(Pw-1) for i in range(0,firstTwo+1,2)]
    else:
        ticks = [i*10**(Pw-1) for i in range(0,firstTwo+1,5)]
    return ticks, [str(i) for i in ticks]

def makeYTicks(N):
    Pw = int(np.log10(N))
    firstOne = int(str(N)[0])
    firstTwo = int(str(N)[:2])
    if firstOne >= 5:
        ticks = [i*10**Pw for i in range(firstOne+1)]
    elif firstTwo <= 20:
        ticks = [i*10**(Pw-1) for i in range(0,firstTwo+1,2)]
    else:
        ticks = [i*10**(Pw-1) for i in range(0,firstTwo+1,5)]
    return [N-1-i for i in ticks], [str(i) for i in ticks]

def bShow(bMap,xS,xE,yS,yE):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    bRMSFull = np.sqrt(cp.sum(bMap**2)/bPixelX/bPixelY/3)
    HB = min(2*bRMSFull, cp.max(bMap))
    LB = max(-2*bRMSFull,cp.min(bMap))
    if (HB>0 and LB<0):
        VM = max(HB,-LB)
        HB = VM
        LB = -VM
    imx = axs[0].imshow(cp.asnumpy(cp.transpose(bMap[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0].set_title('Bx')
    imy = axs[1].imshow(cp.asnumpy(cp.transpose(bMap[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1].set_title('By')
    imz = axs[2].imshow(cp.asnumpy(cp.transpose(bMap[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[2].set_title('Bz')

    for ax in axs:
        ax.axhline(y=bPixelY-1-yS, color='green', linestyle='--', linewidth=1)
        ax.axhline(y=bPixelY-1-yE, color='green', linestyle='--', linewidth=1)
        ax.axvline(x=xS, color='green', linestyle='--', linewidth=1)
        ax.axvline(x=xE, color='green', linestyle='--', linewidth=1)
        ax.set_xlim(0, bPixelX)
        ax.set_ylim(bPixelY, 0)
        
    xTicks, xTickLabels = makeXTicks(bPixelX)
    yTicks, yTickLabels = makeYTicks(bPixelY)

    for i in range(3):
        axs[i].set_xticks(xTicks)
        axs[i].set_yticks(yTicks)
        axs[i].set_xticklabels(xTickLabels)
        axs[i].set_yticklabels(yTickLabels)
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(imz, cax=cbar_ax).set_label('Magnetic Field')

    plt.savefig('bShowTEMP.png', format='png')

    plt.close(fig)

    return

def bCropping():
    global bStartX, bStartY, bEndX, bEndY, bCropX, bCropY, bRMS, bCropped, AGenerated

    if not(bCheckVar.get()):
        messagebox.showinfo("Cropping Error", "B data not loaded!")
        return

    ifBalanced = balanceVar.get()
    
    bShow(bExp,bStartX,bEndX,bStartY,bEndY)

    bCropWindow = tk.Toplevel(app)
    bCropWindow.attributes('-topmost', True)
    bCropWindow.attributes('-topmost', False)
    bCropWindow.title("Selecting B Region")

    canvas = tk.Canvas(bCropWindow)
    BImage = tk.PhotoImage(master=canvas, file='bShowTEMP.png')
    BImageLabel = tk.Label(bCropWindow, image=BImage)
    BImageLabel.image = BImage
    BImageLabel.pack()

    startXLabel = tk.Label(bCropWindow, text="StartX=")
    startXLabel.pack()

    defaultSX=tk.StringVar()
    defaultSX.set(str(bStartX))
    startXEntry = tk.Entry(bCropWindow,textvariable=defaultSX)
    startXEntry.pack()

    startYLabel = tk.Label(bCropWindow, text="StartY=")
    startYLabel.pack()

    defaultSY=tk.StringVar()
    defaultSY.set(str(bStartY))
    startYEntry = tk.Entry(bCropWindow,textvariable=defaultSY)
    startYEntry.pack()

    endXLabel = tk.Label(bCropWindow, text="EndX=")
    endXLabel.pack()

    defaultEX=tk.StringVar()
    defaultEX.set(str(bEndX))
    endXEntry = tk.Entry(bCropWindow,textvariable=defaultEX)
    endXEntry.pack()

    endYLabel = tk.Label(bCropWindow, text="EndY=")
    endYLabel.pack()

    defaultEY=tk.StringVar()
    defaultEY.set(str(bEndY))
    endYEntry = tk.Entry(bCropWindow,textvariable=defaultEY)
    endYEntry.pack()

    def on_enter_button_clicked():
        global bStartX, bStartY, bEndX, bEndY, bCropX, bCropY, bCroppedRMS, bCropped, mAVG, AGenerated

        try:
            nbsx = int(startXEntry.get())
        except:
            nbsx = bStartX
        try:
            nbex = int(endXEntry.get())
        except:
            nbex = bEndX
        try:
            nbsy = int(startYEntry.get())
        except:
            nbsy = bStartY
        try:
            nbey = int(endYEntry.get())
        except:
            nbey = bEndY
        if nbex>bPixelX or nbey>bPixelY or nbsx<0 or nbsy<0:
            messagebox.showinfo("Cropping Failed", "Input value out of B map!")
            bCropWindow.attributes('-topmost', True)
            bCropWindow.attributes('-topmost', False)
            return
        nbcx = nbex - nbsx
        nbcy = nbey - nbsy
        if nbcx<=0 or nbcy<=0:
            messagebox.showinfo("Cropping Failed", "Start value greater than end value!")
            bCropWindow.attributes('-topmost', True)
            bCropWindow.attributes('-topmost', False)
            return
        bStartX = nbsx
        bStartY = nbsy
        bEndX = nbex
        bEndY = nbey
        bCropX = nbcx
        bCropY = nbcy
        bCropped = bExp[bStartX:bEndX,bStartY:bEndY,:]
        if ifBalanced:
            bAVG = cp.mean(bCropped,axis=(0,1)) / bCropX / bCropY
            bCropped = bCropped - bAVG[cp.newaxis,cp.newaxis,:]
        bCroppedRMS = float(cp.sqrt(cp.sum(bCropped**2)/bCropX/bCropY/3))
        croppedRMSLabel.config(text="ROI RMS=%s"%("{:.4e}".format(bCroppedRMS)))
        bCropXLabel.config(text="ROI width=%s"%(bCropX))
        bCropYLabel.config(text="ROI height=%s"%(bCropY))
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        ASizeLabel.config(text="size(est.): %.2fGB"%(72*bCropX*bCropY*NofP/1024/1024/1024))
        AGenerated = False
        ACheckVar.set(0)
        
        bCropWindow.destroy()
        
        return
    
    def on_check_button_clicked():
        try:
            nbsxT = int(startXEntry.get())
        except:
            nbsxT = bStartX
        try:
            nbexT = int(endXEntry.get())
        except:
            nbexT = bEndX
        try:
            nbsyT = int(startYEntry.get())
        except:
            nbsyT = bStartY
        try:
            nbeyT = int(endYEntry.get())
        except:
            nbeyT = bEndY
        bShow(bExp,nbsxT,nbexT,nbsyT,nbeyT)
        BImage = tk.PhotoImage(master=canvas, file='bShowTEMP.png')
        BImageLabel.configure(image=BImage)
        BImageLabel.image = BImage
        os.remove('bShowTEMP.png')
        return
    
    def on_cancel_button_clicked():
        bCropWindow.destroy()
        return
    
    bCropWindow.protocol("WM_DELETE_WINDOW", on_cancel_button_clicked)

    cropFrame = tk.Frame(bCropWindow)
    cropFrame.pack(pady=10,anchor='center')

    cropEnter = tk.Button(cropFrame, text="Enter", command=on_enter_button_clicked)
    cropEnter.grid(row=0,column=0)
    cropCheck = tk.Button(cropFrame, text="Check", command=on_check_button_clicked)
    cropCheck.grid(row=0,column=1)
    cropCancel = tk.Button(cropFrame, text="Cancel", command=on_cancel_button_clicked)
    cropCancel.grid(row=0,column=2)

    os.remove('bShowTEMP.png')
    
    return

cropBButton = tk.Button(loadBFrame2, text="crop ROI", command=bCropping, width=9)
cropBButton.grid(row=0, column=0, padx=(5, 5))

balanceLabel = tk.Label(loadBFrame2, text="balanced?", width=14)
balanceLabel.grid(row=0, column=1, padx=(0, 0))

balanceVar = tk.IntVar(value=0)
balanceCheckBox = tk.Checkbutton(loadBFrame2, variable=balanceVar, state='normal')
balanceCheckBox.grid(row=0, column=1, padx=(86,3))

def balanceInfo():
    messagebox.showinfo("Balanced", "The average values of Bx, By, and Bz will be subtracted on cropping if the 'Balance' checkbox is selected.")
    return

balanceHelp = tk.Button(loadBFrame2, text="?", command=balanceInfo, font=("Arial"), width=1)
balanceHelp.grid(row=0, column=1, padx=(115,6))

bCropXLabel = tk.Label(loadBFrame2, text="ROI width=%s"%(bCropX), width=14)
bCropXLabel.grid(row=0, column=2, padx=(5, 5))

bCropYLabel = tk.Label(loadBFrame2, text="ROI height=%s"%(bCropY), width=14)
bCropYLabel.grid(row=0, column=3, padx=(5, 5))

croppedRMSLabel = tk.Label(loadBFrame2, text="ROI RMS=%s"%(0.0), width=16)
croppedRMSLabel.grid(row=0,column=4,padx=5)

loadBBlank = tk.Label(loadBFrame, text="\n", font=("Helvetica", 16))
loadBBlank.grid(row=0, column=3, rowspan=2, padx=25, pady=5)

###################################
##                               ##
## allocate magnetization region ##
##                               ##
###################################

allocateMFrame = tk.Frame(app)
allocateMFrame.pack(pady=(10,10), anchor='center')

allocateMBlank1 = tk.Label(allocateMFrame, text="\n", width=2)
allocateMBlank1.grid(row=0, column=0, rowspan=2, padx=(0,0), pady=5)

allocateMLabel = tk.Label(allocateMFrame, text="Magnetization\nRegion Allocate", font=("Helvetica", 16))
allocateMLabel.grid(row=0, column=1, rowspan=2, padx=(6, 0), pady=5)

allocateMFrame1 = tk.Frame(allocateMFrame)
allocateMFrame1.grid(row=0, column=2, padx=10)

mStatus = tk.Label(allocateMFrame1, text="region loaded:")
mStatus.grid(row=0, column=0, padx=(5, 0))

mCheckVar = tk.IntVar(value=int(regionLoaded))
mCheckBox = tk.Checkbutton(allocateMFrame1, variable=mCheckVar, state='disabled')
mCheckBox.grid(row=0, column=0, padx=(112,0))

mPixelXLabel = tk.Label(allocateMFrame1, text="M width pixel=%s"%(mPixelX), width=14)
mPixelXLabel.grid(row=0, column=1, padx=(5, 5))

mPixelYLabel = tk.Label(allocateMFrame1, text="M height pixel=%s"%(mPixelY), width=14)
mPixelYLabel.grid(row=0, column=2, padx=(5, 5))

mPixelNLabel = tk.Label(allocateMFrame1, text="region pixel=%s"%(NofP), width=14)
mPixelNLabel.grid(row=0, column=3, padx=(5, 5))

def mRegionShow():

    if not(mCheckVar.get()):
        messagebox.showinfo("Failed to show", "Region not allocated!")
        return
    
    fig, axs = plt.subplots(1,1)

    fig.canvas.manager.set_window_title('magnetization region')

    axs.imshow(np.transpose(weight[:,::-1]), cmap='bwr',vmin=-1,vmax=1)
        
    xTicks, xTickLabels = makeXTicks(mPixelX)
    yTicks, yTickLabels = makeYTicks(mPixelY)

    axs.set_xticks(xTicks)
    axs.set_yticks(yTicks)
    axs.set_xticklabels(xTickLabels)
    axs.set_yticklabels(yTickLabels)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')

    app.iconify()

    plt.show()

    app.deiconify()
    app.attributes("-topmost", True)
    app.attributes("-topmost", False)

    return

regionShowButton = tk.Button(allocateMFrame1,text="show region", command=mRegionShow)
regionShowButton.grid(row=0,column=4,padx=5)

allocateMFrame2 = tk.Frame(allocateMFrame)
allocateMFrame2.grid(row=1, column=2, padx=(4,0))

magnificationLabel = tk.Label(allocateMFrame2,text="magnification=")
magnificationLabel.grid(row=0, column=0, padx=(0,0))

defaultMagnification=tk.StringVar()
defaultMagnification.set("1.0")
magnificationEntry = tk.Entry(allocateMFrame2,textvariable=defaultMagnification,width=6)
magnificationEntry.grid(row=0,column=0, padx=(130,0))

def magnificationInfo():
    messagebox.showinfo("Magnification", "Rescale the selected PNG image by multiplying both the width and height pixel dimensions by the magnification factor.")
    return

magnificationHelp = tk.Button(allocateMFrame2, text="?", command=magnificationInfo, font=("Arial"), width=1)
magnificationHelp.grid(row=0, column=0, padx=(200,6))

def regionAllocate():
    global mPixelX, mPixelY, NofP, listX, listY, mRes, mAVG, weight, regionLoaded, AGenerated

    app.iconify()

    magFactor = magnificationEntry.get()
    magFactor = '1.0' if magFactor=='' else magFactor
    try:
        magFactor = eval(magFactor)
        if magFactor<=0:
            messagebox.showinfo("Value Error", "Invalid magnification!")
            app.deiconify()
            app.attributes("-topmost", True)
            app.attributes("-topmost", False)
            return
    except:
        messagebox.showinfo("Value Error", "Invalid magnification!")
        app.deiconify()
        app.attributes("-topmost", True)
        app.attributes("-topmost", False)
        return

    try:
        image_path = filedialog.askopenfilename(title="Select camera image")
        img_ = cv2.imread(image_path)
        mPixelYT, mPixelXT = img_.shape[0:2]
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()
        messagebox.showinfo("File can not open", "Invalid file path!")
        app.deiconify()
        app.attributes("-topmost", True)
        app.attributes("-topmost", False)
        return
    
    def main():
        global all_polygons, current_polygon, all_polygons_on_image, polygon_on_image, img, img_copy
        all_polygons = []
        current_polygon = []
        all_polygons_on_image = []
        polygon_on_image = []
        def draw_polygons(event, x, y, flags, param):
            global all_polygons, current_polygon, all_polygons_on_image, polygon_on_image, img, img_copy
            if event == cv2.EVENT_LBUTTONDOWN:  
                current_polygon.append((round((x+0.5)*magFactor-0.5), round((y+0.5)*magFactor-0.5)))
                polygon_on_image.append((x,y))
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                if len(current_polygon) > 1:
                    cv2.line(img, polygon_on_image[-2], polygon_on_image[-1], (255, 0, 0), 2)
                cv2.imshow('region allocation', img)
            elif event == cv2.EVENT_RBUTTONDOWN:  
                if len(current_polygon) > 2:
                    cv2.line(img, polygon_on_image[-1], polygon_on_image[0], (255, 0, 0), 2)
                    all_polygons.append(current_polygon.copy())
                    all_polygons_on_image.append(polygon_on_image.copy())
                    current_polygon = []
                    polygon_on_image = []
                    img = img_copy.copy()
                    for polygon in all_polygons_on_image:
                        for i in range(len(polygon)):
                            cv2.circle(img, polygon[i], 2, (0, 255, 0), -1)
                            cv2.line(img, polygon[i], polygon[(i + 1) % len(polygon)], (255, 0, 0), 2)
                    cv2.imshow('region allocation', img)
            return None
        def get_polygon_mask(img_shape, polygons):
            mask = np.zeros((int(magFactor*img_shape[0])+1, int(magFactor*img_shape[1])+1), dtype=np.uint8)
            for vertices in polygons:
                vertices = np.array([vertices], dtype=np.int32)
                cv2.fillPoly(mask, vertices, 1)
            return mask
        def get_pixels_within_polygons(img, polygons):
            mask = get_polygon_mask(img.shape, polygons)
            points = np.column_stack(np.where(mask == 1))
            return points
        
        img = cv2.imread(image_path)
        
        img_copy = img.copy()
        cv2.namedWindow('region allocation')
        cv2.setMouseCallback('region allocation', draw_polygons)
        cv2.imshow('region allocation', img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if cv2.getWindowProperty('region allocation', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        if all_polygons:
            return get_pixels_within_polygons(img, all_polygons)
        else:
            1/0

    while True:
        try:
            selected = main()
            break
        except:
            response=messagebox.askyesno("Allocation failed", "No region selected! Reallocate?")
            if not(response):
                app.deiconify()
                app.attributes("-topmost", True)
                app.attributes("-topmost", False)
                return

    app.deiconify()
    app.attributes("-topmost", True)
    app.attributes("-topmost", False)
    
    listX, listY = [],[]
    mPixelX, mPixelY = (int(magFactor*mPixelXT)+1, int(magFactor*mPixelYT)+1)
    selected_N = list(set((y,mPixelY-1-x) for x,y in selected))
    NofP = len(selected_N)
    weight = np.zeros(shape=(mPixelX,mPixelY))
    for i in range(NofP):
        listX.append(selected_N[i][0])
        listY.append(selected_N[i][1])
        weight[listX[-1],listY[-1]] = 1
    listX, listY = np.array(listX), np.array(listY)
    mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
    mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
    mPixelXLabel.config(text="M width pixel=%s"%(mPixelX), width=14)
    mPixelYLabel.config(text="M height pixel=%s"%(mPixelY), width=14)
    mPixelNLabel.config(text="region pixel=%s"%(NofP), width=14)
    regionLoaded = True
    mCheckVar.set(value=int(regionLoaded))
    ASizeLabel.config(text="size(est.): %.2fGB"%(72*bCropX*bCropY*NofP/1024/1024/1024))
    AGenerated = False
    ACheckVar.set(0)

    return

selectPNGButton = tk.Button(allocateMFrame2,text="select a PNG", command=regionAllocate)
selectPNGButton.grid(row=0, column=1, padx=(5,8))

def regionLoad():
    global mPixelX, mPixelY, NofP, listX, listY, mRes, mAVG, weight, regionLoaded, AGenerated
    
    image_path = filedialog.askopenfilename(title="Select weight matrix")

    try:
        weightT = np.load(image_path)
        if len(weightT.shape)!=2:
            messagebox.showinfo("Loading Error", "Region NPY file must be a 2D array with 1 representing region only")
            return
        elif not(np.any(weightT == 1)):
            messagebox.showinfo("Loading Error", "Region NPY file must contain at least one pixel with value of 1")
            return
        else:
            weight=weightT
    except:
        messagebox.showinfo("Loading Error", "Failed to load selected file or file selection canceled")
        return
    points = np.column_stack(np.where(weight == 1))
    listX, listY = [],[]
    mPixelX, mPixelY = weight.shape
    NofP = len(points)
    for i in range(NofP):
        listX.append(points[i][0])
        listY.append(points[i][1])
    listX, listY = np.array(listX), np.array(listY)
    mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
    mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
    mPixelXLabel.config(text="M width pixel=%s"%(mPixelX), width=14)
    mPixelYLabel.config(text="M height pixel=%s"%(mPixelY), width=14)
    mPixelNLabel.config(text="region pixel=%s"%(NofP), width=14)
    regionLoaded = True
    mCheckVar.set(value=int(regionLoaded))
    ASizeLabel.config(text="size(est.): %.2fGB"%(72*bCropX*bCropY*NofP/1024/1024/1024))
    AGenerated = False
    ACheckVar.set(0)

    return

selectNPYButton = tk.Button(allocateMFrame2,text="select an NPY", command=regionLoad)
selectNPYButton.grid(row=0, column=2, padx=(8,15))

def regionSaveNPY():

    filename = filedialog.asksaveasfilename(title="Save Region as NPY",
    defaultextension=".npy", 
    filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])

    try:
        np.save(filename,weight)
    except:
        pass
    
    return

saveNPYButton = tk.Button(allocateMFrame2,text="save as NPY", command=regionSaveNPY)
saveNPYButton.grid(row=0, column=3, padx=(15,5))

def regionSavePNG():

    filename = filedialog.asksaveasfilename(title="Save Region as PNG",
    defaultextension=".png", 
    filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

    img = Image.fromarray((np.transpose(weight[:,::-1]) * 255).astype('uint8'), mode='L')
    try:
        img.save(filename)
    except:
        pass
    
    return

savePNGButton = tk.Button(allocateMFrame2,text="save as PNG", command=regionSavePNG)
savePNGButton.grid(row=0, column=4, padx=(5,50))

allocateMBlank2 = tk.Label(allocateMFrame, text="\n", width=7)
allocateMBlank2.grid(row=0, column=3, rowspan=2, padx=(0,0), pady=5)

#################################
##                             ##
## coefficient matrix generate ##
##                             ##
#################################

matrixAFrame = tk.Frame(app)
matrixAFrame.pack(pady=(10,10),anchor='center')

matrixALabel = tk.Label(matrixAFrame, text="Coef Matrix\nGenerate", font=("Helvetica", 16))
matrixALabel.grid(row=0, column=0, rowspan=2, padx=(6, 0), pady=5)

matrixABlank1 = tk.Label(matrixAFrame,text='',width=4)
matrixABlank1.grid(row=0, column=1, rowspan=2, padx=(0, 0), pady=5)

@jit(nopython=True, parallel=True)
def matrixAGenerate_(bCropX,bCropY,NofP,width,height,bPixelX,bPixelY,mPixelX,mPixelY,bStartX,bStartY,xShift,yShift,distance,listX,listY):
    
    rPosition = np.zeros((bCropX,bCropY,NofP,3),dtype = np.float64)
    for i in prange(bCropX):
        for j in prange(bCropY):
            for k in prange(NofP):
                bX = (bStartX+i+0.5) * width / bPixelX
                bY = (bStartY+j+0.5) * height / bPixelY
                mX = (listX[k]+0.5) * width / mPixelX + xShift
                mY = (listY[k]+0.5) * height / mPixelY + yShift
                rPosition[i,j,k,0] = bX - mX
                rPosition[i,j,k,1] = bY - mY
                rPosition[i,j,k,2] = np.sqrt(rPosition[i,j,k,0]**2 + rPosition[i,j,k,1]**2 + distance**2)
    
    matrixAnp = np.zeros(shape=(bCropX,bCropY,NofP,3,3),dtype=np.float64)
    for i in prange(bCropX):
        for j in prange(bCropY):
            for k in prange(NofP):
                rPos2_5 = rPosition[i, j, k, 2]**5
                rPos2_3 = rPosition[i, j, k, 2]**3
                rPosX = rPosition[i, j, k, 0]
                rPosY = rPosition[i, j, k, 1]

                matrixAnp[i, j, k, 0, 0] = 3 * rPosX * rPosX / rPos2_5 - 1 / rPos2_3
                matrixAnp[i, j, k, 0, 1] = 3 * rPosX * rPosY / rPos2_5
                matrixAnp[i, j, k, 0, 2] = -3 * rPosX * distance / rPos2_5
                matrixAnp[i, j, k, 1, 0] = matrixAnp[i, j, k, 0, 1]
                matrixAnp[i, j, k, 1, 1] = 3 * rPosY * rPosY / rPos2_5 - 1 / rPos2_3
                matrixAnp[i, j, k, 1, 2] = -3 * rPosY * distance / rPos2_5
                matrixAnp[i, j, k, 2, 0] = matrixAnp[i, j, k, 0, 2]
                matrixAnp[i, j, k, 2, 1] = matrixAnp[i, j, k, 1, 2]
                matrixAnp[i, j, k, 2, 2] = 3 * distance * distance / rPos2_5 - 1 / rPos2_3

    #matrixA = cp.asarray(matrixAnp)
    #del rPosition

    return matrixAnp

@hideWindow
def matrixAGenerate():
    global matrixA, AGenerated

    if not(bLoaded and regionLoaded):
        messagebox.showinfo("Matrix not generated","Load B and allocate region first!")
        return
    
    timeStart = time()

    matrixAnp = matrixAGenerate_(bCropX=bCropX,
                                 bCropY=bCropY,
                                 NofP=NofP,
                                 width=width,
                                 height=height,
                                 bPixelX=bPixelX,
                                 bPixelY=bPixelY,
                                 mPixelX=mPixelX,
                                 mPixelY=mPixelY,
                                 bStartX=bStartX,
                                 bStartY=bStartY,
                                 xShift=xShift,
                                 yShift=yShift,
                                 distance=distance,
                                 listX=listX,
                                 listY=listY)
    matrixA = cp.asarray(matrixAnp*1e-7)
    del matrixAnp

    timeStop = time()

    AGenerated = True
    ACheckVar.set(value=1)
    if (timeStop-timeStart)<60:
        ATimeLabel.config(text='time used: %ss'%(int(timeStop-timeStart)))
    else:
        ATimeLabel.config(text='time used: %sm%ss'%(int(timeStop-timeStart)//60,int(timeStop-timeStart)%60))

    return

matrixAFrame1 = tk.Frame(matrixAFrame)
matrixAFrame1.grid(row=0,column=2,padx=(15,5))

matrixAGenerateButton = tk.Button(matrixAFrame1,text='generate',command=matrixAGenerate)
matrixAGenerateButton.grid(row=0,column=0,padx=(5,0))

AStatus = tk.Label(matrixAFrame1, text="matrix generated:", width=17)
AStatus.grid(row=0, column=1, padx=(0, 0))

ACheckVar = tk.IntVar(value=int(AGenerated))
ACheckBox = tk.Checkbutton(matrixAFrame1, variable=ACheckVar, state='disabled')
ACheckBox.grid(row=0, column=1, padx=(124,0))

ATimeLabel = tk.Label(matrixAFrame1,text='time used:',width=16)
ATimeLabel.grid(row=0,column=2,padx=(5,5))

ASizeLabel = tk.Label(matrixAFrame1,text="size(est.): %.2fGB"%(0),width=16)
ASizeLabel.grid(row=0,column=3,padx=(5,5))

matrixSizeWarningLabel = tk.Label(matrixAFrame,text="Reduce ROI and decrease the number of magnetization pixels for a smaller matrix size.")
matrixSizeWarningLabel.grid(row=1,column=2,padx=(0,0))

matrixABlank2 = tk.Label(matrixAFrame,text='',width=1)
matrixABlank2.grid(row=0, column=3, rowspan=2, padx=(0, 0), pady=5)

memoryWarningFrame = tk.LabelFrame(matrixAFrame, bd=1, relief='solid')
memoryWarningFrame.grid(row=0, column=4, rowspan=2, padx=(6, 0), pady=5)

memoryWarning = tk.Label(memoryWarningFrame,text='Memory may leak!\nReboot to release!',font=(12))
memoryWarning.pack()

##############
##          ##
## OPTIMIZE ##
##          ##
##############

runFrame = tk.Frame(app)
runFrame.pack(pady=(10,1),anchor='center')

runLabel = tk.Label(runFrame, text="Run Optimization", font=("Helvetica", 16))
runLabel.grid(row=0, column=0, rowspan=1, padx=(6, 0), pady=2)

runBlank1 = tk.Label(runFrame, text="", width=5)
runBlank1.grid(row=0, column=1, padx=(0, 0), pady=2)

def setParameter():
    global fromZero, pars, rate, dist

    setParWindow = tk.Toplevel(app)
    setParWindow.attributes('-topmost', True)
    setParWindow.attributes('-topmost', False)
    setParWindow.title("Setting optimization parameters")
    setParWindow.geometry(f"360x320+{position_right+30}+{position_top+30}")

    def setParClose():
        setParWindow.destroy()
        return
    
    setParWindow.protocol("WM_DELETE_WINDOW", setParClose)

    fromZeroFrame = tk.Frame(setParWindow)
    fromZeroFrame.pack(pady=5)

    fromZeroLabel = tk.Label(fromZeroFrame,text="from zero?")
    fromZeroLabel.grid(row=0, column=0,padx=(5,1),pady=5)

    fromZeroCheckVar = tk.IntVar(value=int(fromZero))
    fromZeroCheckBox = tk.Checkbutton(fromZeroFrame,variable=fromZeroCheckVar)
    fromZeroCheckBox.grid(row=0, column=1,padx=(1,1),pady=5)

    def fromZeroHelp():
        messagebox.showinfo("From zero","If from zero checkbox is on, the next optimization will start from 0. Otherwise, it will start from current reconstructed M.")
        setParWindow.attributes('-topmost', True)
        setParWindow.attributes('-topmost', False)
        return
    
    fromZeroHelpButton = tk.Button(fromZeroFrame, text="?", command=fromZeroHelp, font=("Arial"), width=1)
    fromZeroHelpButton.grid(row=0,column=2,padx=(1,5),pady=5)

    gradFrame = tk.Frame(setParWindow)
    gradFrame.pack(pady=5)

    rateLabel = tk.Label(gradFrame,text="rate=")
    rateLabel.grid(row=0,column=0,padx=(5,1),pady=5)

    rateString = tk.StringVar()
    rateString.set(str(rate))
    rateEntry = tk.Entry(gradFrame,textvariable=rateString,width=7)
    rateEntry.grid(row=0,column=1,padx=(1,5),pady=5)
    
    distLabel = tk.Label(gradFrame,text="dist=")
    distLabel.grid(row=0,column=2,padx=(5,1),pady=5)

    distString = tk.StringVar()
    distString.set(str(dist))
    distEntry = tk.Entry(gradFrame,textvariable=distString,width=7)
    distEntry.grid(row=0,column=3,padx=(1,2),pady=5)

    def gradHelp():
        messagebox.showinfo("Gradient parameters",
                            """The code using both first and second order derivative for optimization.
                            
'Rate' determines how long one optimization step is.
0 means not moving and if the function is perfectly quadratic,
1 will stop at minimum, <1 value will stop before, and >1 value will stop after.
                            
'Dist' determines how long one step is to find the gradient.""")
        setParWindow.attributes('-topmost', True)
        setParWindow.attributes('-topmost', False)
        return
    
    gradHelpButton = tk.Button(gradFrame, text="?", command=gradHelp, font=("Arial"), width=1)
    gradHelpButton.grid(row=0,column=4,padx=(2,5),pady=1)

    parInputFrame1 = tk.Frame(setParWindow)
    parInputFrame1.pack(side='top',anchor='w',padx=1,pady=(5,1))

    parInputLabel = tk.Label(parInputFrame1,text="optimization parameter input:")
    parInputLabel.grid(row=0,column=0,padx=5,pady=1)

    parInputFrame2 = tk.Frame(setParWindow)
    parInputFrame2.pack(side='top',anchor='w',padx=1,pady=1)

    parInputBlank1 = tk.Label(parInputFrame2,text='',width=3)
    parInputBlank1.grid(row=0,column=0,padx=1,pady=1)

    parFormatLabel1 = tk.Label(parInputFrame2,text='Format:')
    parFormatLabel1.grid(row=0,column=1,padx=1,pady=1)

    parInputFrame3 = tk.Frame(setParWindow)
    parInputFrame3.pack(side='top',anchor='w',padx=1,pady=(1,1))

    parInputBlank2 = tk.Label(parInputFrame3,text='',width=6)
    parInputBlank2.grid(row=0,column=0,padx=1,pady=1)

    parFormatLabel2 = tk.Label(parInputFrame3,text="alpha,beta,gamma,Q,repeat,threshold,maxRound")
    parFormatLabel2.grid(row=0,column=1,padx=1,pady=1)
    
    def parFormatHelp():
        messagebox.showinfo("Optimization parameters",
                            """Alpha penalizes unsmoothness: larger alpha leads to smoother magnetization.
Beta penalizes large value: larger beta suppresses extreme large value.
Gamma penalizes difference from quantum charge Q.

"Repeat" determines repeat time in one round,
if the loss decreases within threshold (relatively) after one round, optimization stops.
If maxRound is hit without convergence, optimization will also stop.""")
        setParWindow.attributes('-topmost', True)
        setParWindow.attributes('-topmost', False)
        return

    parFormatHelpButton = tk.Button(parInputFrame3, text="?", command=parFormatHelp, font=("Arial"), width=1)
    parFormatHelpButton.grid(row=0,column=2,padx=1,pady=1)

    parFormatLabel3 = tk.Label(setParWindow,text="Add more parameter sets by adding new lines")
    parFormatLabel3.pack(side='top',anchor='center',pady=(1,3))

    parEntryFrame = tk.Frame(setParWindow)
    parEntryFrame.pack(side='top',fill=tk.BOTH,expand=True)

    parString = tk.StringVar()
    parString.set(str(pars))
    parEntry = tk.Text(parEntryFrame,height=4,wrap="word",width=40)
    parEntry.pack(side=tk.LEFT,expand=True)
    parEntry.insert(tk.END, parString.get())
    parEntryScrollBar = tk.Scrollbar(parEntryFrame,command=parEntry.yview, width=20)
    parEntryScrollBar.pack(side=tk.RIGHT, fill=tk.Y)
    parEntry.config(yscrollcommand=parEntryScrollBar.set)
    parEntryFrame.pack_propagate(False)

    setParConfirmFrame = tk.Frame(setParWindow)
    setParConfirmFrame.pack(side="top",anchor='center',pady=5)

    def setParConfirm():
        global fromZero, pars, rate, dist

        ## check validation before replacing

        try:
            rateT = eval(rateEntry.get())
            if rateT<=0:
                messagebox.showinfo("Value error","Invalid rate!")
                setParWindow.attributes('-topmost', True)
                setParWindow.attributes('-topmost', False)
                return
        except:
            messagebox.showinfo("Value error","Invalid rate!")
            setParWindow.attributes('-topmost', True)
            setParWindow.attributes('-topmost', False)
            return
        try:
            distT = eval(distEntry.get())
            if distT<=0:
                messagebox.showinfo("Value error","Invalid dist!")
                setParWindow.attributes('-topmost', True)
                setParWindow.attributes('-topmost', False)
                return
        except:
            messagebox.showinfo("Value error","Invalid dist!")
            setParWindow.attributes('-topmost', True)
            setParWindow.attributes('-topmost', False)
            return
        parT = parEntry.get("1.0", tk.END).strip()
        parList = [item for item in parT.split('\n') if item!='']
        try:
            for parLine in parList:
                a,b,g,q,rp,th,mr = tuple([eval(item) for item in re.split(r',\s*', parLine) if item!=''])
                if a<0 or b<0 or g<0 or int(rp)<=0 or th<0 or int(mr)<0:
                    messagebox.showinfo("Value error","Invalid optimization parameters!")
                    setParWindow.attributes('-topmost', True)
                    setParWindow.attributes('-topmost', False)
                    return
        except:
            messagebox.showinfo("Value error","Invalid optimization parameters!")
            setParWindow.attributes('-topmost', True)
            setParWindow.attributes('-topmost', False)
            return

        fromZero = bool(fromZeroCheckVar.get())
        pars = parT
        rate = rateT
        dist = distT

        setParWindow.destroy()

        return

    setParConfirmButton = tk.Button(setParConfirmFrame,text='Enter',command=setParConfirm,width=6)
    setParConfirmButton.grid(row=0,column=0,padx=5,pady=5)

    setParCancelButton = tk.Button(setParConfirmFrame,text='Cancel',command=setParClose,width=6)
    setParCancelButton.grid(row=0,column=1,padx=5,pady=5)

    return

setParButton = tk.Button(runFrame,text="set parameter",command=setParameter,width=11)
setParButton.grid(row=0,column=2,padx=(5,12),pady=2)

def topoCharge(M):
    mM_norms = cp.linalg.norm(M, axis=2)
    mM_norms = cp.where(mM_norms == 0, 1, mM_norms)
    mM_n = M / mM_norms[:,:,cp.newaxis]
    dM_dx = cp.roll(mM_n, shift=-1, axis=0)
    dM_dy = cp.roll(mM_n, shift=-1, axis=1)
    cross_product = cp.cross(dM_dx, dM_dy)
    dot_product = cp.einsum('ijk,ijk->ij', mM_n, cross_product)
    integral = cp.sum(dot_product)
    Qc = integral / 4 / cp.pi
    return Qc

def lossF(mRes,alpha,beta,gamma):
    
    bRes = cp.tensordot(matrixA,mRes,axes=([2,4],[0,1]))
    result = cp.sum((bRes - bCropped) ** 2) / bCroppedRMS**2 / bCropX / bCropY / 3
    del bRes

    if alpha != 0:
        mRes_ = cp.zeros(shape=(mPixelX,mPixelY,3),dtype=cp.float64)
        for i in range(NofP):
            mRes_[listX[i],listY[i],:] = mRes[i,:]
        xGrad = cp.roll(mRes_, shift=-1, axis=0)
        yGrad = cp.roll(mRes_, shift=-1, axis=1)
        result += (cp.sum((xGrad-mRes_)**2) + cp.sum((yGrad-mRes_)**2)) / 2 / mAVG**2 / NofP * alpha

    if beta != 0:
        result += cp.sum(mRes**2) / mAVG**2 / NofP * beta

    if gamma[0] != 0:
        if alpha == 0:
            mRes_ = cp.zeros(shape=(mPixelX,mPixelY,3),dtype=cp.float64)
            for i in range(NofP):
                mRes_[listX[i],listY[i],:] = mRes[i,:]
        result += gamma[0] * (gamma[1]-topoCharge(mRes_))**2

    return result

def step(mRes, alpha, beta, gamma, rate, dist):
    dirc_ = cp.random.choice([-1,-1,-1,-1,1,1,1,1,1],size=mRes.shape)
    decay_ = dist * mAVG
    delta_ = decay_ * dirc_

    vp = lossF(mRes + delta_, alpha, beta, gamma)
    vm = lossF(mRes - delta_, alpha, beta, gamma)
    v_ = lossF(mRes, alpha, beta, gamma)
    F_ = (vp - vm) / 2 / dist
    F__ = (vp + vm - 2*v_) / dist**2

    return mRes - rate * F_/F__ * mAVG * dirc_

def saveOptimizationData(pathname,cSet,setLen):
    ## save M
    mMat = cp.zeros(shape=(mPixelX,mPixelY,3),dtype=cp.float64)
    for i in range(NofP):
        mMat[listX[i],listY[i],:] = mRes[i,:]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title('reconstructed magnetization')
    
    mRMSFull = np.sqrt(cp.sum(mRes**2)/NofP/3)
    HB = min(2*mRMSFull, cp.max(mRes))
    LB = max(-2*mRMSFull,cp.min(mRes))
    if (HB>0 and LB<0):
        VM = max(HB,-LB)
        HB = VM
        LB = -VM
    imx = axs[0].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0].set_title('Mx')
    imy = axs[1].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1].set_title('My')
    imz = axs[2].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[2].set_title('Mz')
    
    xTicks, xTickLabels = makeXTicks(mPixelX)
    yTicks, yTickLabels = makeYTicks(mPixelY)

    for i in range(3):
        axs[i].set_xticks(xTicks)
        axs[i].set_yticks(yTicks)
        axs[i].set_xticklabels(xTickLabels)
        axs[i].set_yticklabels(yTickLabels)
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')

    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    fig.colorbar(imx, cax=cbar_ax).set_label('Magnetization')

    plt.savefig(os.path.join(os.path.join(pathname, "Set%s"%(str(cSet).zfill(setLen))),"M.png"), format='png')
    cp.save(os.path.join(os.path.join(pathname, "Set%s"%(str(cSet).zfill(setLen))),"M.npy"),mMat)
    plt.close(fig)

    ## save B
    bRes = cp.tensordot(matrixA,mRes,axes=([2,4],[0,1]))

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5)
    fig.canvas.manager.set_window_title('reconstructed magnetic field')
    
    bRMSCropped = np.sqrt(cp.sum(bCropped**2)/bCropX/bCropY/3)
    HB = min(2*bRMSCropped, cp.max(bCropped))
    LB = max(-2*bRMSCropped,cp.min(bCropped))
    if (HB>0 and LB<0):
        VM = max(HB,-LB)
        HB = VM
        LB = -VM
    imxr = axs[0][0].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][0].set_title('Bx Reconstructed')
    imyr = axs[0][1].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][1].set_title('By Reconstructed')
    imzr = axs[0][2].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][2].set_title('Bz Reconstructed')

    imx = axs[1][0].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][0].set_title('Bx Experiment')
    imy = axs[1][1].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][1].set_title('By Experiment')
    imz = axs[1][2].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][2].set_title('Bz Experiment')
        
    xTicks, xTickLabels = makeXTicks(bCropX)
    yTicks, yTickLabels = makeYTicks(bCropY)

    for i in range(6):
        axs[i//3][i%3].set_xticks(xTicks)
        axs[i//3][i%3].set_yticks(yTicks)
        axs[i//3][i%3].set_xticklabels(xTickLabels)
        axs[i//3][i%3].set_yticklabels(yTickLabels)
        axs[i//3][i%3].set_xlabel('X')
        axs[i//3][i%3].set_ylabel('Y')

    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    fig.colorbar(imx, cax=cbar_ax).set_label('Magnetic Field')
    
    plt.savefig(os.path.join(os.path.join(pathname, "Set%s"%(str(cSet).zfill(setLen))),"B.png"), format='png')
    cp.save(os.path.join(os.path.join(pathname, "Set%s"%(str(cSet).zfill(setLen))),"B.npy"),bRes)
    plt.close(fig)

    return

@hideWindow
def stepFull():
    global mRes, gLoss

    if not(AGenerated):
        messagebox.showinfo("Optimization Error","Coefficient matrix not generated!")
        return
    
    responce = messagebox.askokcancel("Run Optimization","Be sure to set optimization parameters as needed before running experiment!")

    if not(responce):
        return
    
    writeLog = runLogCheckVar.get()
    saveSet = saveSetCheckVar.get()
    saveRound = saveRoundCheckVar.get()
    
    if writeLog:
        filename = filedialog.asksaveasfilename(title="Save Run Log",
defaultextension=".txt", 
filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            logFile = open(filename,'w')
        else:
            messagebox.showinfo("Saving Error","Run log file selection canceled!")
            runLogCheckVar.set(0)
            return
    
    if saveSet:
        pathname = filedialog.askdirectory(title="Select Folder to Save Reconstructed Results")
        if pathname == '':
            messagebox.showinfo("Saving Error","Folder selection canceled!")
            saveSetCheckVar.set(0)
            disableSaveRound()
            return
    
    timeStart = time()

    if writeLog:
        logFile.write("Optimization starting at %s\n\n"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        logFile.write("Physical parameters:\n\twidth=%s, height=%s, distance=%s, xShift=%s, yShift=%s\n\n"%(width,height,distance,xShift,yShift))
        logFile.write("ROI:\n\tX: %s ~ %s\n\tY: %s ~ %s\n\n"%(bStartX,bEndX,bStartY,bEndY))
        logFile.write("Optimization starting from zero.\n\n" if fromZero else "Optimization not starting from zero\n\n")
        logFile.write("Rate = %s\nDist = %s\n\n"%(rate,dist))
        logFile.write("Optimization parameters:\n%s\n"%(pars))
        logFile.close()

    if fromZero:
        mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)

    cSet = -1
    parList = [item for item in pars.split('\n') if item!='']
    parL = [tuple([eval(item) for item in re.split(r',\s*', parLine) if item!='']) for parLine in parList]
    if saveSet:
        setLen = len(str(len(parL)-1))
    for alpha,beta,gamma,Qc,repeat,thresh,maxRound in parL:
        setStart = time()
        repeat, maxRound = int(repeat), int(maxRound)
        cSet += 1
        count = 0
        lLoss = 1.
        runTimeLabel.config(text="Running set %s"%cSet)
        if writeLog:
            logFile = open(filename,'a')
            logFile.write("\nOptimization set %s:\n"%cSet)
            logFile.write("\talpha=%s, beta=%s, gamma=%s, Q=%s, repeat=%s, threshold=%s, maxRound=%s\n"%(alpha,beta,gamma,Qc,repeat,thresh,maxRound))
            logFile.close()
        if saveSet:
            os.makedirs(os.path.join(pathname, "Set%s"%(str(cSet).zfill(setLen))), exist_ok=True)
        while True:
            if count>=maxRound:
                if writeLog:
                    logFile = open(filename,'a')
                    setStop = time()
                    logFile.write("\tSet %s finished by hitting the maxRound.\n"%cSet)
                    setTime = int(setStop - setStart)
                    logFile.write("\tSet time used: %sm%ss\n"%(setTime//60,setTime%60))
                    logFile.write("\tLoss after current set: %s\n"%(cLoss))
                    logFile.close()
                if saveSet:
                    saveOptimizationData(pathname,cSet,setLen)
                break
            for j in range(repeat):
                mRes = step(mRes,alpha,beta,(gamma,Qc),rate,dist)

            if saveRound:
                saveOptimizationData(pathname,cSet,setLen)

            count += 1
            cLoss = lossF(mRes,0,0,(0,0))
            if ((lLoss-cLoss)/lLoss <= thresh) and (lLoss>cLoss):
                if writeLog:
                    logFile = open(filename,'a')
                    setStop = time()
                    logFile.write("\tSet %s finished after repeat round %s.\n"%(cSet,count))
                    setTime = int(setStop - setStart)
                    logFile.write("\tSet time used: %sm%ss\n"%(setTime//60,setTime%60))
                    logFile.write("\tLoss after current set: %s\n"%(cLoss))
                    logFile.close()
                if saveSet:
                    saveOptimizationData(pathname,cSet,setLen)
                break
            lLoss = cLoss

    gLoss = cLoss
    timeStop = time()
    timeInSec = int(timeStop-timeStart)
    lossLabel.config(text="loss: %.4e"%(gLoss))
    if timeInSec//3600 == 0:
        runTimeLabel.config(text="time used: %sm%s"%(timeInSec//60,timeInSec%60))
    else:
        timeInMin = timeInSec//60
        runTimeLabel.config(text="time used: %sh%sm"%(timeInMin//60,timeInMin%60))

    if writeLog:
        logFile = open(filename,'a')
        logFile.write("\nOptimization finished\n")
        logFile.write("Loss: %s\n"%(gLoss))
        if timeInSec//3600 == 0:
            logFile.write("Total time used: %sm%s"%(timeInSec//60,timeInSec%60))
        else:
            logFile.write("Total time used: %sh%sm"%(timeInMin//60,timeInMin%60))
        logFile.close()

    return

runButton = tk.Button(runFrame,text='RUN',command=stepFull,width=11)
runButton.grid(row=0,column=3,padx=(12,0),pady=2)

runLogLabel = tk.Label(runFrame,text="save log?")
runLogLabel.grid(row=0,column=4,padx=(5,0),pady=2)

runLogCheckVar = tk.IntVar(value=1)
runLogCheckBox = tk.Checkbutton(runFrame, variable=runLogCheckVar, state='normal')
runLogCheckBox.grid(row=0, column=4, padx=(84,0),pady=2)

runTimeLabel = tk.Label(runFrame,text='time used:',width=16)
runTimeLabel.grid(row=0,column=5,padx=(5,5),pady=2)

lossLabel = tk.Label(runFrame,text="loss:",width=12)
lossLabel.grid(row=0,column=6,padx=(5,5),pady=2)

runBlank2 = tk.Label(runFrame, text="", width=7)
runBlank2.grid(row=0, column=7, padx=(0, 0), pady=2)

########################
##                    ##
## SHOW & SAVE RESULT ##
##                    ##
########################

showFrame = tk.Frame(app)
showFrame.pack(pady=(2,10),anchor='center')

showLabel = tk.Label(showFrame,text="Show & Save", font=("Helvetica", 16))
showLabel.grid(row=0,column=0,padx=(5,5),pady=2)

showBlank1 = tk.Label(showFrame,text='',width=7)
showBlank1.grid(row=0,column=1,padx=(5,5),pady=2)

saveResultFrame = tk.Frame(showFrame)
saveResultFrame.grid(row=0,column=2,padx=1,pady=2)

saveSetLabel = tk.Label(saveResultFrame,text="save each set?")
saveSetLabel.grid(row=0,column=0,padx=1,pady=2)

def disableSaveRound():
    saveRoundCheckVar.set(value=0)
    if saveSetCheckVar.get():
        saveRoundCheckBox.config(state=tk.NORMAL)
    else:
        saveRoundCheckBox.config(state=tk.DISABLED)
    return

saveSetCheckVar = tk.IntVar(value=0)
saveSetCheckBox = tk.Checkbutton(saveResultFrame, variable=saveSetCheckVar, state='normal',command=disableSaveRound)
saveSetCheckBox.grid(row=0, column=1, padx=(1,3),pady=2)

saveRoundLabel = tk.Label(saveResultFrame,text="save after round?")
saveRoundLabel.grid(row=0,column=2,padx=(3,1),pady=2)

saveRoundCheckVar = tk.IntVar(value=0)
saveRoundCheckBox = tk.Checkbutton(saveResultFrame, variable=saveRoundCheckVar, state='disabled')
saveRoundCheckBox.grid(row=0, column=3, padx=(1,1),pady=2)

def saveInfo():
    messagebox.showinfo("Save Result",
                        """If you want to save your results after each parameter set (one line in the optimization parameters), enable the 'save each set' checkbox. The program will save each set in separate folders. It is recommended to select the folder that contains the run log.

The reconstructed magnetization data and reconstructed magnetic field data will be saved in both .npy and .png formats.

If 'save after round' is enabled, the program will create or update (and replace) the saved results after each repeat round, ensuring your data is preserved in case of an unexpected exit.

If you only want the final result, IT IS RECOMMENDED TO SAVE THE RESULTS AFTER THE OPTIMIZATION USING THE 'SHOW' AND 'SAVE' BUTTONS.""")
    return

saveHelp = tk.Button(saveResultFrame, text="?", command=saveInfo, font=("Arial"), width=1)
saveHelp.grid(row=0, column=4, padx=(1,9),pady=2)

def showM():

    if not(AGenerated):
        messagebox.showinfo("See, you're getting worked up again","No data to show!")
        return
    
    mMat = cp.zeros(shape=(mPixelX,mPixelY,3),dtype=cp.float64)
    for i in range(NofP):
        mMat[listX[i],listY[i],:] = mRes[i,:]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title('reconstructed magnetization')
    
    mRMSFull = np.sqrt(cp.sum(mRes**2)/NofP/3)
    HB = min(2*mRMSFull, cp.max(mRes))
    LB = max(-2*mRMSFull,cp.min(mRes))
    if (HB>0 and LB<0):
        VM = max(HB,-LB)
        HB = VM
        LB = -VM
    imx = axs[0].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0].set_title('Mx')
    imy = axs[1].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1].set_title('My')
    imz = axs[2].imshow(cp.asnumpy(cp.transpose(mMat[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[2].set_title('Mz')
    
    xTicks, xTickLabels = makeXTicks(mPixelX)
    yTicks, yTickLabels = makeYTicks(mPixelY)

    for i in range(3):
        axs[i].set_xticks(xTicks)
        axs[i].set_yticks(yTicks)
        axs[i].set_xticklabels(xTickLabels)
        axs[i].set_yticklabels(yTickLabels)
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')

    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    fig.colorbar(imx, cax=cbar_ax).set_label('Magnetization')

    plt.show()

    return

showMButton = tk.Button(showFrame,text="show M",command=showM,width=7)
showMButton.grid(row=0,column=3,padx=(9,5),pady=2)

def showB():

    if not(AGenerated):
        messagebox.showinfo("See, you're getting worked up again","No data to show!")
        return
    
    bRes = cp.tensordot(matrixA,mRes,axes=([2,4],[0,1]))

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5)
    fig.canvas.manager.set_window_title('reconstructed magnetic field')
    
    bRMSCropped = np.sqrt(cp.sum(bCropped**2)/bCropX/bCropY/3)
    HB = min(2*bRMSCropped, cp.max(bCropped))
    LB = max(-2*bRMSCropped,cp.min(bCropped))
    if (HB>0 and LB<0):
        VM = max(HB,-LB)
        HB = VM
        LB = -VM
    imxr = axs[0][0].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][0].set_title('Bx Reconstructed')
    imyr = axs[0][1].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][1].set_title('By Reconstructed')
    imzr = axs[0][2].imshow(cp.asnumpy(cp.transpose(bRes[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[0][2].set_title('Bz Reconstructed')

    imx = axs[1][0].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,0])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][0].set_title('Bx Experiment')
    imy = axs[1][1].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,1])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][1].set_title('By Experiment')
    imz = axs[1][2].imshow(cp.asnumpy(cp.transpose(bCropped[:,::-1,2])), cmap='bwr',vmin=LB,vmax=HB)
    axs[1][2].set_title('Bz Experiment')
        
    xTicks, xTickLabels = makeXTicks(bCropX)
    yTicks, yTickLabels = makeYTicks(bCropY)

    for i in range(6):
        axs[i//3][i%3].set_xticks(xTicks)
        axs[i//3][i%3].set_yticks(yTicks)
        axs[i//3][i%3].set_xticklabels(xTickLabels)
        axs[i//3][i%3].set_yticklabels(yTickLabels)
        axs[i//3][i%3].set_xlabel('X')
        axs[i//3][i%3].set_ylabel('Y')
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    fig.colorbar(imxr, cax=cbar_ax).set_label('Magnetic Field')

    plt.show()

    return None

showBButton = tk.Button(showFrame,text="show B",command=showB,width=7)
showBButton.grid(row=0,column=4,padx=(5,7),pady=2)

def saveM():

    if not(AGenerated):
        messagebox.showinfo("See, you're getting worked up again","No data to save!")
        return

    mMat = cp.zeros(shape=(mPixelX,mPixelY,3),dtype=cp.float64)
    for i in range(NofP):
        mMat[listX[i],listY[i],:] = mRes[i,:]

    filename = filedialog.asksaveasfilename(title="Save Magnetization",
    defaultextension=".npy", 
    filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])

    try:
        np.save(filename,mMat)
    except:
        pass
    
    return

saveMButton = tk.Button(showFrame,text="save M",command=saveM,width=7)
saveMButton.grid(row=0,column=5,padx=(7,5),pady=2)

def saveB():

    if not(AGenerated):
        messagebox.showinfo("See, you're getting worked up again","No data to save!")
        return

    bRes = cp.tensordot(matrixA,mRes,axes=([2,4],[0,1]))

    filename = filedialog.asksaveasfilename(title="Save Magnetic Field",
    defaultextension=".npy", 
    filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])

    try:
        np.save(filename,bRes)
    except:
        pass
    
    return

saveBButton = tk.Button(showFrame,text="save B",command=saveB,width=7)
saveBButton.grid(row=0,column=6,padx=5,pady=2)

showBlank2 = tk.Label(showFrame,text='',width=3)
showBlank2.grid(row=0,column=7,padx=(5,5),pady=2)

##################################
##                              ##
## PHYSICAL SCALE PARAMETER FIT ##
##                              ##
##################################

parFitFrame = tk.Frame(app)
parFitFrame.pack(pady=(10,4),anchor='center')

parFitLabel = tk.Label(parFitFrame,text="Phys-Scale Fit", font=("Helvetica", 16))
parFitLabel.grid(row=0,column=0,padx=(5,5),pady=2)

parFitBlank1 = tk.Label(parFitFrame,text='',width=6)
parFitBlank1.grid(row=0,column=1,padx=2,pady=2)

def parFitInfo():
    messagebox.showinfo("Parameter Setting Warning","""Physical scale fitting also use optimization function, and it share optimization parameters (except fromZero) with optimization function.
                        
To change optimization parameters, use "set parameter" button at "Run Optimization" section.
To make the optimization stable, A THRESHOLD VALUE OF 0 IS RECOMMENDED.
                        
Fitting time will be 3*fittingRound times longer than the optimization time on average!
IT IS REASONABLE FOR WAITING FOR A WHILE.""")
    return

parFitHelp = tk.Button(parFitFrame,text='!',font=("Arial"),command=parFitInfo)
parFitHelp.grid(row=0,column=1,padx=(30,3),pady=2)

def setFitParameter():
    global fitD, fitX, fitY, distance, xShift, yShift, AGenerated, fitRate, fitDist, mAVG, fitRound

    setFitWindow = tk.Toplevel(app)
    setFitWindow.attributes('-topmost', True)
    setFitWindow.attributes('-topmost', False)
    setFitWindow.title("Setting physical scale fitting parameters")
    setFitWindow.geometry(f"360x310+{position_right+30}+{position_top+30}")

    def setFitClose():
        setFitWindow.destroy()
        return
    
    setFitWindow.protocol("WM_DELETE_WINDOW", setFitClose)

    gradFitFrame = tk.Frame(setFitWindow)
    gradFitFrame.pack(pady=5)

    rateFitLabel = tk.Label(gradFitFrame,text="rate=")
    rateFitLabel.grid(row=0,column=0,padx=(5,1),pady=5)

    rateFitString = tk.StringVar()
    rateFitString.set(str(fitRate))
    rateFitEntry = tk.Entry(gradFitFrame,textvariable=rateFitString,width=7)
    rateFitEntry.grid(row=0,column=1,padx=(1,5),pady=5)
    
    distFitLabel = tk.Label(gradFitFrame,text="dist=")
    distFitLabel.grid(row=0,column=2,padx=(5,1),pady=5)

    distFitString = tk.StringVar()
    distFitString.set(str(fitDist))
    distFitEntry = tk.Entry(gradFitFrame,textvariable=distFitString,width=7)
    distFitEntry.grid(row=0,column=3,padx=(1,2),pady=5)

    def gradFitHelp():
        messagebox.showinfo("Gradient parameters",
                            """The code using both first and second order derivative for optimization.
                            
'Rate' determines how long one optimization step is.
0 means not moving and if the function is perfectly quadratic,
1 will stop at minimum, <1 value will stop before, and >1 value will stop after.
                            
'Dist' determines how long one step is to find the gradient.""")
        setFitWindow.attributes('-topmost', True)
        setFitWindow.attributes('-topmost', False)
        return
    
    gradFitHelpButton = tk.Button(gradFitFrame, text="?", command=gradFitHelp, font=("Arial"), width=1)
    gradFitHelpButton.grid(row=0,column=4,padx=(2,5),pady=5)

    fitWhatFrame = tk.Frame(setFitWindow)
    fitWhatFrame.pack(side='top',anchor='w',padx=1,pady=(5,5))

    fitWhatBlank = tk.Label(fitWhatFrame,text="",width=2)
    fitWhatBlank.grid(row=0,column=0,padx=5,pady=5)

    fitWhatLabel = tk.Label(fitWhatFrame,text="Fit?")
    fitWhatLabel.grid(row=0,column=1,padx=5,pady=5)
    
    def fitWhatInfo():
        messagebox.showinfo("Fitting Options",
                            """Choose which parameter will change during fitting.
Set initial value in the input box.
DO NOT SET ZERO IF YOU WANT IT TO CHANGE!""")
        setFitWindow.attributes('-topmost', True)
        setFitWindow.attributes('-topmost', False)
        return

    fitWhatHelp = tk.Button(fitWhatFrame, text="?", command=fitWhatInfo, font=("Arial"), width=1)
    fitWhatHelp.grid(row=0,column=2,padx=1,pady=5)

    fitInputFrame = tk.Frame(setFitWindow)
    fitInputFrame.pack(side="top",anchor='center',padx=1,pady=5)

    fitDLabel = tk.Label(fitInputFrame,text="distance=",anchor='e',width=9)
    fitDLabel.grid(row=0,column=0,padx=2,pady=5)
    fitDString = tk.StringVar()
    fitDString.set(str(distance))
    fitDEntry = tk.Entry(fitInputFrame,textvariable=fitDString,width=7)
    fitDEntry.grid(row=0,column=1,padx=2,pady=5)
    fitDCheckVar = tk.IntVar(value=int(fitD))
    fitDCheckBox = tk.Checkbutton(fitInputFrame,variable=fitDCheckVar)
    fitDCheckBox.grid(row=0,column=2,padx=2,pady=5)

    fitXLabel = tk.Label(fitInputFrame,text="xShift=",anchor='e',width=9)
    fitXLabel.grid(row=1,column=0,padx=2,pady=5)
    fitXString = tk.StringVar()
    fitXString.set(str(xShift))
    fitXEntry = tk.Entry(fitInputFrame,textvariable=fitXString,width=7)
    fitXEntry.grid(row=1,column=1,padx=2,pady=5)
    fitXCheckVar = tk.IntVar(value=int(fitX))
    fitXCheckBox = tk.Checkbutton(fitInputFrame,variable=fitXCheckVar)
    fitXCheckBox.grid(row=1,column=2,padx=2,pady=5)

    fitYLabel = tk.Label(fitInputFrame,text="yShift=",anchor='e',width=9)
    fitYLabel.grid(row=2,column=0,padx=2,pady=5)
    fitYString = tk.StringVar()
    fitYString.set(str(yShift))
    fitYEntry = tk.Entry(fitInputFrame,textvariable=fitYString,width=7)
    fitYEntry.grid(row=2,column=1,padx=2,pady=5)
    fitYCheckVar = tk.IntVar(value=int(fitY))
    fitYCheckBox = tk.Checkbutton(fitInputFrame,variable=fitYCheckVar)
    fitYCheckBox.grid(row=2,column=2,padx=2,pady=5)

    fitRoundFrame = tk.Frame(setFitWindow)
    fitRoundFrame.pack(side="top",anchor='center',padx=1,pady=5)

    fitRoundLabel = tk.Label(fitRoundFrame,text="fitting round =")
    fitRoundLabel.pack(side="left",padx=1,pady=5)

    fitRoundString = tk.StringVar()
    fitRoundString.set(str(fitRound))
    fitRoundEntry = tk.Entry(fitRoundFrame,textvariable=fitRoundString,width=7)
    fitRoundEntry.pack(side="left",padx=1,pady=5)

    setFitConfirmFrame = tk.Frame(setFitWindow)
    setFitConfirmFrame.pack(side="top",anchor='center',pady=5)

    def setFitConfirm():
        global fitD, fitX, fitY, distance, xShift, yShift, AGenerated, fitRate, fitDist, mAVG, fitRound

        ## check validation before replacing

        try:
            rateT = eval(rateFitEntry.get())
            if rateT<=0:
                messagebox.showinfo("Value error","Invalid rate!")
                setFitWindow.attributes('-topmost', True)
                setFitWindow.attributes('-topmost', False)
                return
        except:
            messagebox.showinfo("Value error","Invalid rate!")
            setFitWindow.attributes('-topmost', True)
            setFitWindow.attributes('-topmost', False)
            return
        try:
            distT = eval(distFitEntry.get())
            if distT<=0:
                messagebox.showinfo("Value error","Invalid dist!")
                setFitWindow.attributes('-topmost', True)
                setFitWindow.attributes('-topmost', False)
                return
        except:
            messagebox.showinfo("Value error","Invalid dist!")
            setFitWindow.attributes('-topmost', True)
            setFitWindow.attributes('-topmost', False)
            return
        try:
            DT = np.float64(fitDEntry.get())
            DX = np.float64(fitXEntry.get())
            DY = np.float64(fitYEntry.get())
        except:
            messagebox.showinfo("Value error","Invalid physical scale input!")
            setFitWindow.attributes('-topmost', True)
            setFitWindow.attributes('-topmost', False)
            return
        try:
            fitRoundT = int(fitRoundEntry.get())
            if fitRoundT<=0:
                messagebox.showinfo("Value error","Invalid fitting round input!")
                setFitWindow.attributes('-topmost', True)
                setFitWindow.attributes('-topmost', False)
                return
        except:
            messagebox.showinfo("Value error","Invalid fitting round input!")
            setFitWindow.attributes('-topmost', True)
            setFitWindow.attributes('-topmost', False)
            return

        fitD = bool(fitDCheckVar.get())
        fitX = bool(fitXCheckVar.get())
        fitY = bool(fitYCheckVar.get())
        distance = DT
        xShift = DX
        yShift = DY
        AGenerated = False
        fitRate = rateT
        fitDist = distT
        fitRound = fitRoundT
        distanceLabel.config(text="distance=%s"%(str(distance) if len(str(distance))<=7 else "{:.1e}".format(distance)))
        xShiftLabel.config(text="xShift=%s"%(str(xShift) if len(str(xShift))<=9 else "{:.3e}".format(xShift)))
        yShiftLabel.config(text="yShift=%s"%(str(yShift) if len(str(yShift))<=9 else "{:.3e}".format(yShift)))
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        ACheckVar.set(0)

        setFitWindow.destroy()

        return

    setFitConfirmButton = tk.Button(setFitConfirmFrame,text='Enter',command=setFitConfirm,width=6)
    setFitConfirmButton.grid(row=0,column=0,padx=5,pady=5)

    setFitCancelButton = tk.Button(setFitConfirmFrame,text='Cancel',command=setFitClose,width=6)
    setFitCancelButton.grid(row=0,column=1,padx=5,pady=5)

    return

setFitButton = tk.Button(parFitFrame,text="set parameter",command=setFitParameter,width=11)
setFitButton.grid(row=0,column=2,padx=(5,12),pady=2)

@hideWindow
def parFit():
    global mRes, gLoss, distance, xShift, yShift, matrixA, mAVG, AGenerated

    if not(bLoaded and regionLoaded):
        messagebox.showinfo("Fitting Error","B data not loaded or region not allocated!")
        return
    
    responce = messagebox.askokcancel("Run Fitting","Be sure to set fitting and optimization parameters as needed before running experiment!")

    if not(responce):
        return

    writeFitLog = runLogCheckVar.get()
    
    if writeFitLog:
        filename = filedialog.asksaveasfilename(title="Save Fitting Log",
defaultextension=".txt", 
filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            logFile = open(filename,'w')
        else:
            messagebox.showinfo("Saving Error","Fitting log file selection canceled!")
            runLogCheckVar.set(0)
            return
    
    timeStart = time()
    AGenerated = False
    ACheckVar.set(0)

    if writeFitLog:
        logFile.write("Physical scale fitting starting at %s\n\n"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        logFile.write("Physical parameters:\n\twidth=%s, height=%s\n\n"%(width,height))
        logFile.write("ROI:\n\tX: %s ~ %s\n\tY: %s ~ %s\n\n"%(bStartX,bEndX,bStartY,bEndY))
        logFile.write("Rate = %s\nDist = %s\n\n"%(fitRate,fitDist))
        logFile.write("Fit distance = %s\nFit xShift   = %s\nFit yShift   = %s\n\n"%(fitD,fitX,fitY))
        logFile.write("Optimization parameters:\n%s\n\n"%(pars))
        logFile.write("Optimization Rate = %s\nOptimization Dist = %s\n\n"%(rate,dist))
        logFile.close()
    
    for i in range(fitRound):
        roundStart = time()
        d_, x_, y_ = distance, xShift, yShift
        while True:
            delta_ = np.float64(fitDist)*np.random.choice([-1,-2,-2,-1,-2,3,1,1,3,1,0],size=3)
            if not(fitD):
                delta_[0]=0
            if not(fitX):
                delta_[1]=0
            if not(fitY):
                delta_[2]=0
            if np.any(delta_ != 0):
                break
        
        pl = (delta_+1)
        distance, xShift, yShift = d_*pl[0], x_*pl[1], y_*pl[2]
        matrixAnp = matrixAGenerate_(bCropX=bCropX,
                                 bCropY=bCropY,
                                 NofP=NofP,
                                 width=width,
                                 height=height,
                                 bPixelX=bPixelX,
                                 bPixelY=bPixelY,
                                 mPixelX=mPixelX,
                                 mPixelY=mPixelY,
                                 bStartX=bStartX,
                                 bStartY=bStartY,
                                 xShift=xShift,
                                 yShift=yShift,
                                 distance=distance,
                                 listX=listX,
                                 listY=listY)
        matrixA = cp.asarray(matrixAnp*1e-7)
        del matrixAnp
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
        cSet = -1
        parList = [item for item in pars.split('\n') if item!='']
        parL = [tuple([eval(item) for item in re.split(r',\s*', parLine) if item!='']) for parLine in parList]
        for alpha,beta,gamma,Qc,repeat,thresh,maxRound in parL:
            repeat, maxRound = int(repeat), int(maxRound)
            cSet += 1
            count = 0
            lLoss = 1.
            while True:
                if count>=maxRound:
                    break
                for j in range(repeat):
                    mRes = step(mRes,alpha,beta,(gamma,Qc),rate,dist)
                count += 1
                cLoss = lossF(mRes,0,0,(0,0))
                if ((lLoss-cLoss)/lLoss <= thresh) and (lLoss>cLoss):
                    break
                lLoss = cLoss
        vp = lossF(mRes,0,0,(0,0))

        distance, xShift, yShift = d_, x_, y_
        matrixAnp = matrixAGenerate_(bCropX=bCropX,
                                 bCropY=bCropY,
                                 NofP=NofP,
                                 width=width,
                                 height=height,
                                 bPixelX=bPixelX,
                                 bPixelY=bPixelY,
                                 mPixelX=mPixelX,
                                 mPixelY=mPixelY,
                                 bStartX=bStartX,
                                 bStartY=bStartY,
                                 xShift=xShift,
                                 yShift=yShift,
                                 distance=distance,
                                 listX=listX,
                                 listY=listY)
        matrixA = cp.asarray(matrixAnp*1e-7)
        del matrixAnp
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
        cSet = -1
        parList = [item for item in pars.split('\n') if item!='']
        parL = [tuple([eval(item) for item in re.split(r',\s*', parLine) if item!='']) for parLine in parList]
        for alpha,beta,gamma,Qc,repeat,thresh,maxRound in parL:
            repeat, maxRound = int(repeat), int(maxRound)
            cSet += 1
            count = 0
            lLoss = 1.
            while True:
                if count>=maxRound:
                    break
                for j in range(repeat):
                    mRes = step(mRes,alpha,beta,(gamma,Qc),rate,dist)
                count += 1
                cLoss = lossF(mRes,0,0,(0,0))
                if ((lLoss-cLoss)/lLoss <= thresh) and (lLoss>cLoss):
                    break
                lLoss = cLoss
        v_ = lossF(mRes,0,0,(0,0))
        if writeFitLog:
            logFile = open(filename,'a')
            logFile.write('Round=%s, distance=%s, xShift=%s, yShift=%s, loss=%s\n'%(i,distance,xShift,yShift,v_))
            logFile.close()

        pl = (1-delta_)
        distance, xShift, yShift = d_*pl[0], x_*pl[1], y_*pl[2]
        matrixAnp = matrixAGenerate_(bCropX=bCropX,
                                 bCropY=bCropY,
                                 NofP=NofP,
                                 width=width,
                                 height=height,
                                 bPixelX=bPixelX,
                                 bPixelY=bPixelY,
                                 mPixelX=mPixelX,
                                 mPixelY=mPixelY,
                                 bStartX=bStartX,
                                 bStartY=bStartY,
                                 xShift=xShift,
                                 yShift=yShift,
                                 distance=distance,
                                 listX=listX,
                                 listY=listY)
        matrixA = cp.asarray(matrixAnp*1e-7)
        del matrixAnp
        mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
        mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
        cSet = -1
        parList = [item for item in pars.split('\n') if item!='']
        parL = [tuple([eval(item) for item in re.split(r',\s*', parLine) if item!='']) for parLine in parList]
        for alpha,beta,gamma,Qc,repeat,thresh,maxRound in parL:
            repeat, maxRound = int(repeat), int(maxRound)
            cSet += 1
            count = 0
            lLoss = 1.
            while True:
                if count>=maxRound:
                    break
                for j in range(repeat):
                    mRes = step(mRes,alpha,beta,(gamma,Qc),rate,dist)
                count += 1
                cLoss = lossF(mRes,0,0,(0,0))
                if ((lLoss-cLoss)/lLoss <= thresh) and (lLoss>cLoss):
                    break
                lLoss = cLoss
        vm = lossF(mRes,0,0,(0,0))  
        
        F_ = (vp - vm) / 2 / fitDist
        F__ = (vp + vm - 2*v_) / fitDist**2

        if F__>0:
            pl = 1 - fitRate * np.float64(F_/F__ / fitDist) * delta_
        elif F__<0:
            pl = 1 + 0.5 * fitRate * np.float64(F_/F__ / fitDist) * delta_
        else:
            pl = [1,1,1]
        distance, xShift, yShift = d_*pl[0], x_*pl[1], y_*pl[2]

        roundStop = time()
        if writeFitLog:
            logFile = open(filename,'a')
            if (roundStop-roundStart)<=3600:
                logFile.write('\tround %s time:%sm%ss\n\n'%(i, int(roundStop-roundStart)//60, int(roundStop-roundStart)%60))
            else:
                logFile.write('\tround %s time:%sh%sm\n\n'%(i, (int(roundStop-roundStart)//60)//60, (int(roundStop-roundStart)//60)%60))
            logFile.close()
    
    distanceLabel.config(text="distance=%s"%(str(distance) if len(str(distance))<=7 else "{:.1e}".format(distance)))
    xShiftLabel.config(text="xShift=%s"%(str(xShift) if len(str(xShift))<=9 else "{:.3e}".format(xShift)))
    yShiftLabel.config(text="yShift=%s"%(str(yShift) if len(str(yShift))<=9 else "{:.3e}".format(yShift)))
    matrixAnp = matrixAGenerate_(bCropX=bCropX,
                                 bCropY=bCropY,
                                 NofP=NofP,
                                 width=width,
                                 height=height,
                                 bPixelX=bPixelX,
                                 bPixelY=bPixelY,
                                 mPixelX=mPixelX,
                                 mPixelY=mPixelY,
                                 bStartX=bStartX,
                                 bStartY=bStartY,
                                 xShift=xShift,
                                 yShift=yShift,
                                 distance=distance,
                                 listX=listX,
                                 listY=listY)
    matrixA = cp.asarray(matrixAnp*1e-7)
    del matrixAnp
    AGenerated = True
    ACheckVar.set(1)
    mAVG = bCroppedRMS * distance**3 * 2e7 / NofP
    mRes = cp.zeros(shape=(NofP,3),dtype=cp.float64)
    cSet = -1
    parList = [item for item in pars.split('\n') if item!='']
    parL = [tuple([eval(item) for item in re.split(r',\s*', parLine) if item!='']) for parLine in parList]
    for alpha,beta,gamma,Qc,repeat,thresh,maxRound in parL:
        repeat, maxRound = int(repeat), int(maxRound)
        cSet += 1
        count = 0
        lLoss = 1.
        while True:
            if count>=maxRound:
                break
            for j in range(repeat):
                mRes = step(mRes,alpha,beta,(gamma,Qc),rate,dist)
            count += 1
            cLoss = lossF(mRes,0,0,(0,0))
            if ((lLoss-cLoss)/lLoss <= thresh) and (lLoss>cLoss):
                break
            lLoss = cLoss
    v_ = lossF(mRes,0,0,(0,0))
    gLoss = v_
    lossLabel.config(text="loss: %.4e"%v_)
    lossFitLabel.config(text="loss: %.4e"%v_)
    timeStop = time()
    if writeFitLog:
        logFile = open(filename,'a')
        logFile.write('Fitting finished.\n\tdistance=%s, xShift=%s, yShift=%s\n\tloss=%s\n'%(distance,xShift,yShift,v_))
        if (timeStop-timeStart)<=3600:
            logFile.write('\tround %s time:%sm%ss'%(i, int(timeStop-timeStart)//60, int(timeStop-timeStart)%60))
        else:
            logFile.write('\tround %s time:%sh%sm'%(i, (int(timeStop-timeStart)//60)//60, (int(timeStop-timeStart)//60)%60))
        logFile.close()
    if (timeStop-timeStart)<=3600:
        runFitTimeLabel.config(text='time used: %sm%ss'%(int(timeStop-timeStart)//60, int(timeStop-timeStart)%60))
    else:
        runFitTimeLabel.config(text='time used: %sh%sm'%((int(timeStop-timeStart)//60)//60, (int(timeStop-timeStart)//60)%60))

    return

runFitButton = tk.Button(parFitFrame,text='RUN',command=parFit,width=11)
runFitButton.grid(row=0,column=3,padx=(12,0),pady=2)

runFitLogLabel = tk.Label(parFitFrame,text="save log?")
runFitLogLabel.grid(row=0,column=4,padx=(5,0),pady=2)

runFitLogCheckVar = tk.IntVar(value=1)
runFitLogCheckBox = tk.Checkbutton(parFitFrame, variable=runFitLogCheckVar, state='normal')
runFitLogCheckBox.grid(row=0, column=4, padx=(84,0),pady=2)

runFitTimeLabel = tk.Label(parFitFrame,text='time used:',width=16)
runFitTimeLabel.grid(row=0,column=5,padx=(5,5),pady=2)

lossFitLabel = tk.Label(parFitFrame,text="loss:",width=12)
lossFitLabel.grid(row=0,column=6,padx=(5,5),pady=2)

parFitBlank2 = tk.Label(parFitFrame, text="", width=6)
parFitBlank2.grid(row=0, column=7, padx=(0, 0), pady=2)

##########
##      ##
## MORE ##
##      ##
##########

moreFrame = tk.Frame(app)
moreFrame.pack(pady=(9,8),anchor='center')

def openLink():
    webbrowser.open_new("https://github.com/ycy-ycy/MagRec")
    return

websiteButton = tk.Button(moreFrame,text="visit project website",command=openLink,height=2,width=15)
websiteButton.grid(row=0,column=0,rowspan=2,padx=5,pady=2)

def garbageCollect():

    memoryBefore = psutil.Process(os.getpid()).memory_info().rss
    aDel = gc.collect()
    memoryAfter = psutil.Process(os.getpid()).memory_info().rss
    messagebox.showinfo("Garbage Collected","%s objects deallocated.\nMemory released: %.2fMB"%(aDel,(memoryBefore-memoryAfter)/1024/1024))

    return

garbageCollectButton = tk.Button(moreFrame,text="garbage collect",command=garbageCollect,height=2,width=15)
garbageCollectButton.grid(row=0,column=1,rowspan=2,padx=5,pady=2)

app.mainloop()