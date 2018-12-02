# Project Title

Predicting your fluxes using the Dustpedia database


### Description

We use the learning tools provided by python in order to predict fluxes at a provided wavelength using a supervised learning approach. 
The scripts make use of the multi-wavelength photometry published by the Dustpedia collaboration. 
The table (csv) is available in this directory (https://github.com/maudgalametz/Dustpedia)
and on the Dustpedia collaboration website (http://dustpedia.astro.noa.gr/Photometry)

We are now using the Linear Regression Algorithm. 
Other algorithms are currently tested.

We split the dataset into two, 80\% is used to train our models and 20\% 
are used as a validation dataset. The R^2 regression score function is also
returned by the function.





### Prerequisites


To run the PredictDustpedia.py script, you will need the 5 key libraries scipy, numpy, matplotlib, pandas and sklearn
To install sklearn on your machine, you can install it via pip:
```
pip install -U scikit-learn
```

or conda:
```
conda install scikit-learn
```

To run the Graphic Interface, you will need to have Tkinter installed on your machine

Tkinter is not a pip package, so you might want to use:
```apt-get install python-tk
```






## How to use the script PredictDustpedia.py:

**Inputs:** 

wave: array containing your input wavelength (in microns)

flux: array containing your input fluxes (in Jy)

wavereq: Wavelength of the flux to estimate

**Output** 

Predicted fluxes at wavereq (in Jy)

**Example** 

PredictDustpedia([100,250],[2.8,2.8],350)


## How to use the graphic Interface GraphicInterface.py:

Opens a window where the previous inputs can be manually manipulated

**Output** 

Provide the predicted fluxes at wavereq (in Jy) 

**Example**

execfile('GraphicInterface.py')



## Versioning

We use python 2.7.12. 


## Authors

* **Maud Galametz** 

