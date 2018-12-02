from Tkinter import *
import numpy as np
import ast


# Store the arrays entered in the interface and process them
def Save():
	w = Wavelength.get()
	f = Flux.get()
	wp = Wavelengthp.get()
	wave = np.array(ast.literal_eval(w))
	flux = np.array(ast.literal_eval(f))
	wavep = float(wp)
	print(wave)
	print(flux)
	print(wavep)
	Result = PredictDustpedia(wave,flux,wavep)
	Field4.delete(first=0,last=22)
	Field4.insert(END, Result)


# Create the main window
MyWindow = Tk()
MyWindow.configure(background="#a1dbcd")
MyWindow.title('Predict your flux with Dustpedia')

# Create the Label widgets
Label0 = Label(MyWindow, text = "Please, enter your input values", font=("Helvetica", 16), bg ='#a1dbcd')
Label1 = Label(MyWindow, text = u"Input wavelengths (\u03bcm)",background="#a1dbcd") #"Input wavelengths (\u03bcm) ")
#Label1.pack(side = LEFT, padx = 5, pady = 5)
Label2 = Label(MyWindow, text = 'Input fluxes (Jy) ',background="#a1dbcd")
Label3 = Label(MyWindow, text = u'Wavelength to predict (\u03bcm)',background="#a1dbcd")
Label4 = Label(MyWindow, text = 'Predicted Flux (Jy)',background="#a1dbcd")


# Create the Entry widgets
# Wavelength
Wavelength= StringVar()
Field1 = Entry(MyWindow, textvariable=Wavelength, bg ='#e0ec84', fg='maroon')
Field1.insert(END, '100,250')
Field1.focus_set()
#Field1.pack(side = RIGHT, padx = 5, pady = 5)
# Flux
Flux=StringVar()
Field2 = Entry(MyWindow, textvariable=Flux, bg ='#e0ec84', fg='maroon')
Field2.insert(END, '2.8,2.8')
Field2.focus_set()
# Wavelength to predict
Wavelengthp= StringVar()
Field3 = Entry(MyWindow, textvariable=Wavelengthp, bg ='#e0ec84', fg='maroon')
Field3.insert(END, 350.0)
Field3.focus_set()
# Result predict
Result= StringVar()
Field4 = Entry(MyWindow, textvariable=Result, bg ='#e0ec84', fg='maroon')
#Field3.insert(END, )
#Field3.focus_set()

# Create the Button widget
Button = Button(MyWindow, text ='Validate', relief=RAISED, command = Save, fg="#a1dbcd", bg="#383a39")
#Button.pack(side = LEFT, padx = 5, pady = 5)

Label0.grid(row =1, column=0)
Label1.grid(row =3, sticky =E)
Label2.grid(row =4, sticky =E)
Label3.grid(row =5, sticky =E)
Label4.grid(row =7, sticky =E)

Field1.grid(row =3, column =2)
Field2.grid(row =4, column =2)
Field3.grid(row =5, column =2)
Field4.grid(row =7, column =2)

Button.grid(row =3, column =3, rowspan =3, padx =10, pady =5)

MyWindow.mainloop()
