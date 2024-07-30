import requests
import os
import pandas as pd

### First we will create the folder to contain the climate variables
try:
    os.mkdir("ClimateVariables")
except FileExistsError:
    print("The ClimateVariables folder already exists, please delete or move it and attempt again!")

### Now we need to open the var_names file so we now what variables we want and save them in a list
variables_to_download = pd.read_csv(r"NARR\NARR_var_names.csv", header=None)
varFile = variables_to_download[0].tolist()

### Now we will cycle through each variable and download it from the 
count = 1
for var in varFile:
    url = r"https://downloads.psl.noaa.gov/Datasets/NARR/Monthlies/monolevel/" + var
    r = requests.get(url)
    with open(r"ClimateVariables/" + var, "wb") as code:
        code.write(r.content)
    print(var + " has been downloaded!")
    print(str(count/len(varFile)*100) + "% complete!")
    count += 1