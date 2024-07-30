isoP
================

## Introduction
isoP was orignally developed in December 2014 by C. Delavau as a way of creating gridded precipitation isotopic data. It was developed in matlab which is unfortunately not open source. This is a python version of the original program converted by J. Gray in 2023.
## Installation
In order to run isoP you need to make sure you have downloaded python 3.10 or higher. You can download python from [here](https://www.python.org/downloads/). Once you have downloaded python you need to install the following packages:
* numpy
* pandas
* netCDF4
* scipy
* json
* os
* requests

You can do this by loading up the command prompt and typing the following for each package:
```
pip install [package name]
```
or you can install all the packages at once by typing the following:
```
pip install -r requirements.txt
```
NOTE: The JSON and OS library are included in the standard library so you do not need to install it. If you try running the program and it says the package they are not installled it may have been due to improper installation of python. You may need to reinstall python and ensure that you have the correct version of pip installed.

Afterwards you will need to clone this repository to your system. 

## How to use
### Downloading the program
You can download the program by cloning the repository to your system. This is made simple via the Github website. You can do this by clicking
the green code button and copying the link. You can then open *git bash* and type the following:
```bash
git clone [link]
```
This will download the repository to your system where the git bash is currently located within the directory. This program does not need to be anywhere in particular on your system. You can place it wherever you like.
### Set up of the input
There are two parts to this depending on whether you are running WATFLOOD files or not. If you are using watflood files as inputs you need to make sure you have your SHD files setup within the basin directory. Then you need to make a folder with the main directory called isoP. This is where the program will store the output files as well as a list of coordinates derived from the SHD files. 

If you are not using WATFLOOD files you need to make sure you have setup your directory accordingly. 
```
basinname\isoP\basinname_coords.csv
```
This is where the program will look for the coordinates to use in the program and where it will store the output files. The basin name *does not* matter, only that it is consistent.  

### Running the program
The entirety of the program is run through the command line. You can run the program by opening the command prompt and navigating to the directory where the program is located. You can then run the program by typing the following:
```bash
python isoP.py
```
NOTE: You will be told by the program that you need to download the NARR climate variables first, to do so please select the correct option. This will take sometime as those files are quite large. Once those are downloaded you can then select the option that you want from the main menu. This does not need to be done everytime, but they do need to be updated every now and then. This goes for the teleconnection files as well. However the teleconnection files will have to be done manually as of now.

The program will load up and you can select your options from there. The program will prompt you for the following:
* The path to the main basin directory
* Basin name
* Start year
* End year

## User Profile
The program is equipped with a user profile function that will store the above information into a textfile along with your name and creation date. This is so if you are running the program multiple times you do not have to keep typing in the same information. However, if you would like to change the information you can:
* Delete the user profile file
* Rename the user profile file
* Change the information in the user profile file

Keep an eye out for more prompts that may appear later.

**Note** about the following prompt: 
> "Would you like to account for 18Oppt input uncertainty by calculating prediction intervals? 
NOTE: this is a very time consuming, computationally heavy 
> process. Y/N: "

This pompt **MUST** be answered with a no or n as of this moment. This is because the function has not been finished yet for that part of the program.
# WARNING
This program is still in the beginning stages of development. While it does work it is worth noting that it is not capable of working with UTM coordinates as of yet, so please ensure that you are using latitude and longitude coordinates.

Also, while the python version does currently run with the linear regression models that were used in the orignal matlab version, it lacks the same functionality as those models. Thus there is a JSON file that contains the coefficients and intercepts for the linear regression models.
