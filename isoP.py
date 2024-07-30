# Importing the needed packages that will be used throughout
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, distance
import netCDF4 as nc
import math
import os
import requests
import json
from datetime import datetime, timedelta

def readSHD_File(path, basinName):
    # Converts file into a line by line list
    pathSHD = path + r"\basin" + "\\" + basinName + r"_shd.r2c"
    shdFile = open(pathSHD, "r")
    shdList = shdFile.readlines()
    shdFile.close()
    # JG -- We create an index containing the line # of the SHD file that has 
    # the specific info we need
    info = []
    positionInfo = []
    index=[]
    for k in range(0,len(shdList)):
        if ":SourceFileName" in shdList[k]:
            index.append(k)
        elif ":Projection" in shdList[k]:
            index.append(k)
        elif ":xOrigin" in shdList[k]:
            index.append(k)
        elif ":yOrigin" in shdList[k]:
            index.append(k)
        elif ":xCount" in shdList[k]:
            index.append(k)
        elif ":yCount" in shdList[k]:
            index.append(k)
        elif ":xDelta" in shdList[k]:
            index.append(k)
        elif "yDelta" in shdList[k]:
            index.append(k)
            
    #JG -- Using the nexly created index, we the add the needed info inyo the info list
    for i in index:
        info.append(shdList[i].split())
    #JG -- Separating the actual info from the names we place it into a new list and then 
    # assign them variables
    for j in range(0, len(info)):
        positionInfo.append(info[j][1])
    mapName = positionInfo[0]
    projection = str(positionInfo[1])
    xOrigin = float(positionInfo[2])
    yOrigin = float(positionInfo[3])
    xCount =  int(positionInfo[4])
    yCount = int(positionInfo[5])
    xDelta = float(positionInfo[6])
    yDelta = float(positionInfo[7])

    #--------------------------------------------------------------------------
    #
    # Now that we have the information necessary to calculate the LAT LONG
    # pairs at each WATFLOOD grid. This code below does just that based on the
    # geographic information extracted from the SHP file for your basin.
    #
    #--------------------------------------------------------------------------
    WFcoord_X = np.zeros((yCount+1, xCount+1))
    WFcoord_Y = np.zeros((yCount+1, xCount+1))
    for m in range(yCount, 0, -1):
        for n in range(1, xCount+1, 1):
            if (m==yCount and n==1):
                WFcoord_X[m,n]=(xOrigin+xDelta/2)
                WFcoord_Y[m,n]=(yOrigin+yDelta/2)
            elif (n==1):
                WFcoord_Y[m,n]=WFcoord_Y[m+1, n]+yDelta
            elif (m==yCount):
                WFcoord_X[m,n]=WFcoord_X[m,n-1]+xDelta
            else:
                WFcoord_X[m,n]=WFcoord_X[yCount,n]
                WFcoord_Y[m,n]=WFcoord_Y[m,1]
    #JG -- I am unsure as to why I could not just make the initialiization array smaller
    # regradless, the below code erases the excess rows and columns!
    WFcoord_X = WFcoord_X[1:, 1:]
    WFcoord_Y = WFcoord_Y[1:, 1:]

    #JG -- TBH I am not sure why lines 76-78 exist, but I will copy them anyways
    WFcoord_Y[yCount-1: ] = WFcoord_Y[yCount-1, 0]
    WFcoord_X[:,0] = WFcoord_X[yCount-1, 0]
    numCells = xCount * yCount

    #JG -- These lines reshape the arrays into single column rows containing the information, then combines them
    newCoord_Y = np.reshape(WFcoord_Y, (numCells, 1), 'F')
    newCoord_X = np.reshape(WFcoord_X, (numCells, 1), 'F')
    WFcoords = np.column_stack((newCoord_Y, newCoord_X))
    #--------------------------------------------------------------------------
    #
    # Check if the basin is it LAT LONG coordinates. If it is not, then the LAT
    # LONGS of each grid need to be calculated based on the UTM coords. Prompt 
    # the user for which province the basin is in if this is the case. 
    #
    #--------------------------------------------------------------------------

    #if (projection == "UTM" or projection == "Cartesian"):
        # Will only activate if the projection is a UTM Zone
        #JG -- I can't finish this part of the code yet, MATLAB let you pick the zone like "17T" but python operates differently
        # in order for the process to be more automated it needs to pull the "T" part out on its own, but SHD file doesn't include it

    #JG -- Finally we must create an excel spreadsheet that can be converted into a shape file later
    pathCSV = path + "\\isoP\\" + basinName + "_coords.csv"
    np.savetxt(pathCSV, WFcoords, delimiter=",")

    # at this point, you must use the WF_coords excel file, convert it to a
    # shapefile, and then import that shapefile into ARC GIS software to obtain
    # the Koeppen cliamte classification for the WATFLOOD basin grids. 
    
    print(basinName + "_SHD.r2c file was successfully read in!")

    return WFcoords

def extract_NARR_timeseries(cwd, path, basinName):
    #################################################################
    # This program reads in an array of lat/long coordinates (I.e the
    # basin_coords.xls file previously output by read_SHD_file.m)
    # and extracts the NARR data at the corresponding grid locations.
    #
    #
    # ** IMPORTANT NOTE: If the output files already exist in the folder
    # directory, you must delete them before running this program (ie.
    # NARR_varname_mon_mean). Otherwise the program will just add the extracted
    # data onto the end of the previous dataset.
    #
    #################################################################
    #Extract all NARR data at the WATFLOOD grid coords.
    #This is accomplished through reading in an excel file with the names of
    #the NARR monthly mean files. These files names utilize the naming
    #convention of the NARR repository. Therefore, if the data files need to be
    #updated to include additional years, they can be directly downloaded form
    #the NARR server and the names should still be relevant to this program.
    #This requires reading in an xls file called "filename". This file must be
    #stored in the NARR directory. The PATH vairable below will need to be
    #changed depending on where the isoP file strcuture is stored.

    #From the path name input by the user, backtrack to the main SPL directory
    #to access the isoP_MAIN folder where the NARR netCDF files are stored.
        #JG -- is the main intent with this section of the code just to go back by one 
    #splitPath = path.split("\\")
    #for i in range(0, len(splitPath)-1):
    #    if i == 0:
    #        pathCharm = splitPath[0]
    #    else:
    #        pathCharm = pathCharm + "\\" + splitPath[i]
    
    #Read in the .xls file called 'NARR_var_names' which is stored in the
    #isoP_MAIN\Code folder. This file is a list of all the NARR climate
    #variables to read in. If you don't want to read all of the NARR variablesa
    #in, you can delete them from this list. Or, conversely, if you would like
    #to read more variables in, put the netCDF file for a monthly mean climate
    #varibale in this folder, add it's name to the .xls file and augment this
    #code to read it in.
    pathNARR = cwd + "\\NARR\\NARR_var_names.csv"
    varsNARR = pd.read_csv(pathNARR, header=None)
    numClimatePar = len(varsNARR)
    
    # Read in the WATFLOOD pairs of long/lats (Longitude is in the first column,
    # latitude is in the second column). These were output from read_SHD_file.m
    # # to the SPL\basin\isoP folder earlier.
    pathCoords=path + "\\isoP\\" + basinName + "_coords.csv"
    WFcoords = pd.read_csv(pathCoords, header=None)
    #JG -- Renames the columns to the respective Lat Long pairs
    WFcoords.rename(columns={0: 'LAT', 1: 'LONG'}, inplace=True)

    # Find the number of WATFLOOD points that NARR data needs to be
    # extracted at.
    numPts = len(WFcoords.index)

    # This next portion of code uses latitude and longitude pairs (from in the form [lat,long]) 
    # to find the corresponding NARR grid (xy) to extract the data from.
    # This is accomplished by finding the location of nearest four grid points corresponding to a certain 
    # location consist of latitude and longitude. 
    # NOTE: The NARRlatlon.mat file MUST be in the working directory! Otherwise the code will not run.
    #JG -- This is another instance where the MATLAB features allow this to be done easier, I will find a workaround
    #-- Loading in the NARR lat and lon files
    latPath = cwd + "\\NARR\\" + "NARRlat.csv"
    lonPath = cwd + "\\NARR\\" + "NARRlon.csv" 

    #JG -- Converting the csv to arrays and resizing them so we can combine them
    lat = np.genfromtxt(latPath, delimiter=',', encoding='utf-8-sig')
    lon = np.genfromtxt(lonPath, delimiter=',', encoding='utf-8-sig')
    gridSize = lat.shape
    latr = np.reshape(lat, (np.size(lat), 1), 'F')
    lonr = np.reshape(lon, (np.size(lon), 1), 'F')
    X = np.column_stack((latr, lonr))

    #JG -- Begining nearest neighbour calculations
    delaunayTri = Delaunay(X)
    K = np.zeros((numPts, 1)) #Initialize empty array to hold indices
    xy_NARR = np.zeros((numPts, 2)) #Initialize empty array to hold NN lat lon pairs
    for j in range(numPts):
        coord = WFcoords.iloc[j, :].astype(float)
        dist = distance.cdist(delaunayTri.points, np.array([coord]))
        K[j] = np.argmin(dist) + 1
        gridNo = [K[j] % gridSize[0], (K[j] // gridSize[0])+1]
        xy_NARR[j, 0] = gridNo[1][0]
        xy_NARR[j, 1] = gridNo[0][0] 
    # This loop will cycle through all of the NARR datafiles (ie. one loop for each climate variable)
    # whose names were in the spreadsheet previously read.
    print("Reading in NARR climate variables!")
    output = [[] for _ in range(numClimatePar)]
    for i in range(numClimatePar):
        filetmp = varsNARR.iloc[i, 0]
        ncid = nc.Dataset("ClimateVariables\\" + filetmp, 'r')
        # Depending on which NARR climate variable is being extracted, time (as well as other parts of the data) will
        # be stored in a different dimension of the netCDF file. This portion of
        # code extracts the time variable from the netCDF files, specifying the
        # different locations depending on file name.
        time =  ncid.variables['time'][:]
        timeLength = len(time)

        # Create a matrix of dates for the NARR data (because the format they use is
        # weird and this is just easier to get month and yer next to the
        # corresponding NARR data. If the NARR files are updated to include 2013,
        # the end year will have to be updated to reflect that change as well.
        startYear = 1979
        numYears = timeLength / 12
        numFullYears=math.floor(numYears)
        numPartYears = numYears-numFullYears
        numMonths = round(numPartYears * 12)
        if (timeLength % 12) == 0:
            numCol = timeLength / 12
        else:
            rem = timeLength % 12
            numCol = (timeLength + (12 - rem))/12
        matrixSize = [12, int(numCol)]
        monthMatrix = np.zeros((matrixSize[0], matrixSize[1]))
        yearMatrix = np.zeros((matrixSize[0], matrixSize[1]))
        if (numPartYears == 0):
            numElements = numFullYears * 12
        else:
            numElements = (numFullYears+1)*12

        #JG -- Creating the date arrays to use
        for k in range(numFullYears+1):
            if (k == (numFullYears)):
                for j in range(numMonths):
                    monthMatrix[j,k] = j + 1
                    if k==0:
                        yearMatrix[j,k] = startYear
                    else:
                        yearMatrix[j,k] = (startYear + k)
            else:
                for j in range(12):
                    monthMatrix[j, k] = j +1 
                    if k==0:
                        yearMatrix[j,k] = startYear
                    else:
                        yearMatrix[j,k] = (startYear + k)
        
        #JG -- Fromatting the date arrays
        monthArray = np.reshape(monthMatrix, (numElements, 1), 'F')
        monthArray = monthArray[monthArray != 0]
        yearArray = np.reshape(yearMatrix, (numElements, 1), 'F')
        yearArray = yearArray[yearArray != 0]

        #JG -- Creating the date Matrix
        dateMatrix = np.column_stack((yearArray, monthArray))

        varID = filetmp.split('.')[0]
        #JG -- varID was 6 in matlab code, but I believe it is referring to variable name
        # Extract NARR climate variable time series for each WATFLOOD grid point.
        # NOTE: This data is extracted for the ENTIRE time series (ie. for NARR
        # this is from January 1, 1979 onwards).
        allData = ncid.variables[varID][:]
        out1 = np.zeros((timeLength, numPts))
        for j in range(numPts):
            data = allData[:timeLength, int(xy_NARR[j, 1]), int(xy_NARR[j, 0])]
            out1[:, j] = data
        output[i] = np.concatenate((dateMatrix, out1), axis=1)
    print("NARR climate variables successfully read in for specified WATFLOOD grids!")
    #JG --The final output is off by enough decimals that I am unconfident, however it runs so that will have to wait
    #I beleive the reason for this is likely that the information I am pulling from the netCDF file is different than the matlab version
    return output, pathNARR, #pathCharm

def NARR_format_timeseries_basin(output, pathNARR, startYear, endYear):
    varsNARR = pd.read_csv(pathNARR, header=None)
    varsNARR = varsNARR.to_numpy()
    numClimatePar = len(varsNARR)
    print("Format NARR Climate Variables and crop dataset to specified year.")

    #Initializing some new parameters
    oldNARR= [[] for _ in range(numClimatePar)]
    month = [[] for _ in range(numClimatePar)]
    year = [[] for _ in range(numClimatePar)]
    newNARR = [[] for _ in range(numClimatePar)]

    for k in range(numClimatePar):
        filetmp = varsNARR[k, 1]

        #Read in data from the NARR file
        oldNARR[k] = output[k][:, 2:]
        length, numGrid = oldNARR[k].shape
        #Extracting date into columns of data
        month[k] = output[k][:, 1]
        year[k] = output[k][:, 0]

        #Reinitalizing the newNARR variable (may not keep this)
        newNARR[k] = np.zeros((length, numGrid))

        #Creating a list containg the cumulative variables that need special treatment
        cumVars = ['apcp_mon_mean', 'evap_mon_mean', 'acpcp_mon_mean', 'prwtr_mon_mean']
        if filetmp in cumVars:
            for j in range(numGrid):
                for i in range(length):
                    #Removing negatives from the dataset that may hav ebeen caused by errors in the NARR data
                    #Needed for these variables as negatives are not possible
                    if oldNARR[k][i, j] < 0:
                        oldNARR[k][i, j] = 0
                    
                    #Formating based on 30 day months
                    if (month[k][i] == 4) or (month[k][i] == 6) or (month[k][i] == 9) or (month[k][i] == 11):
                        newNARR[k][i, j] = oldNARR[k][i, j]  * 30
                    #Fromating them depending on February or not and also if a leap year or not
                    elif (month[k][i] == 2):
                        if (year[k][i] % 4 == 0):
                            newNARR[k][i, j] = oldNARR[k][i, j]  * 29
                        else:
                            newNARR[k][i, j] = oldNARR[k][i, j]  * 28
                    #For all other months
                    else:
                        newNARR[k][i, j] = oldNARR[k][i, j]  * 31
        elif filetmp == 'air2m_mon_mean':
            newNARR[k] = oldNARR[k]- 273.15 #Converting from kelvin into celsius
        else:
            newNARR[k] = oldNARR[k]
    #Initializing the rehaped grids
    gridNo = np.zeros((length))
    outputNARR_all = [np.zeros((length, numClimatePar)) for _ in range(numGrid)]
    reshapeNARR = [np.zeros((length, numClimatePar)) for _ in range(numGrid)]

    #Reshaping the grids
    for k in range(numGrid):
        for m in range(numClimatePar):
            reshapeNARR[k][:, m] = newNARR[m][:, k]
            gridNo[:] = k+1
        outputNARR_all[k] = np.column_stack((gridNo, month[0], year[0], reshapeNARR[k]))
    
    #Finding the start row and end row
    newLength = outputNARR_all[0].shape[0]
    #Usuing a for loop to find the start and end years and assign them to the indexs
    for i in range(newLength):
        if (outputNARR_all[0][i, 2] == startYear) and (outputNARR_all[0][i-1, 2] == startYear - 1):
            startIndex = i
        if (outputNARR_all[0][i, 2] == endYear) and (outputNARR_all[0][i+1, 2] == endYear + 1):
            endIndex = i

    #Creating a outputNARR that is cropped to the specified years
    outputNARR = []
    for i in range(len(outputNARR_all)):
        outputNARR.append(outputNARR_all[i][startIndex:endIndex+1, :])
    print("NARR files successfully formatted for specified WATFLOOD grids!")
    return outputNARR

def extract_GIS_info(startYear, endYear, WFcoords, cwd):
    #JG -- Extracting the Lat/Longs from the WFcoords
    numGrids = len(WFcoords)
    lat = WFcoords[:,0]
    lon = WFcoords[:,1]

    #Read in DEM for Canada/Northern Tier of the United States
    print("Extracting GIS information: Elevation, KPN zone indicator.")
    pathDEM = cwd + '\\ModelData\\DEM_CAD.csv'
    dem = np.genfromtxt(pathDEM, delimiter=',', skip_header=1)
    latDEM = dem[:,0]
    lonDEM = dem[:,1]
    altDEM = dem[:,2]

    #Using Delaunay triangulation and nearest neighbour interpolation, find the
    #DEM grid point closest to each WATFLOOD grid point. Extract the elevation
    #from that grid and assign to the corresponding WATFLOOD grid point.
    xDEM = np.concatenate((latDEM[:,None], lonDEM[:,None]), axis=1)

    #Calculate the Delaunay triangulation of the DEM grid points.
    #JG -- Do not ask me how this works, I have no idea. It is a black box.
    tri = Delaunay(xDEM)
    k = np.zeros(numGrids)
    demBasin = np.zeros((numGrids, 3))
    for j in range(numGrids):
        coord = WFcoords[j, :]
        dist = distance.cdist(tri.points, np.array([coord]))
        k[j] = np.argmin(dist)
        demBasin[j,:] = np.hstack((WFcoords[j,:], altDEM[int(k[j])]))
    
    #Based on the 1020 shapefile for Canada/northern US, read in Kpn Zone
    pathKPN = cwd + '\\ModelData\\Kpn_zone.csv'
    kpnZone = np.genfromtxt(pathKPN, delimiter=',', skip_header=1)
    kpn = kpnZone[:,2]
    latKPN = kpnZone[:,0]
    lonKPN = kpnZone[:,1]

    #Same procedure as before using Delaunay triangulation and nearest neighbour interpolation
    # to find the KPN zone for each WATFLOOD grid point. Extract kpn zone from that grid and assign to the 
    # corresponding WATFLOOD grid.
    xKPN = np.concatenate((latKPN[:,None], lonKPN[:,None]), axis=1)
    tri = Delaunay(xKPN)
    k = np.zeros(numGrids)
    kpnBasin = np.zeros((numGrids, 3))
    for j in range(numGrids):
        coord = WFcoords[j, :]
        dist = distance.cdist(tri.points, np.array([coord]))
        k[j] = np.argmin(dist)
        kpnBasin[j,:] = np.hstack((WFcoords[j,:], kpn[int(k[j])]))

    #Combining the elevation and KPN zone data into one array
    length = (endYear-startYear+1)*12
    dataGEO = [np.zeros((length, 4)) for _ in range(numGrids)]

    for k in range(numGrids):
        dataGEO[k][:length, 0] = lat[k]
        dataGEO[k][:length, 1] = lon[k]
        dataGEO[k][:length, 2] = demBasin[k,2]
        dataGEO[k][:length, 3] = kpnBasin[k,2]


    print("GIS information successfully read in for specified WATFLOOD grids!")
    return dataGEO

def extract_tele_timeseries_basin(WFcoords, cwd, startYear, endYear):
    #Find the number of watflood points that TELE data needs to be extracted at
    numGrids = len(WFcoords)
    pathTele = cwd + "\\Tele\\index_files.csv"

    #Read in the teleconnection index data
    teleFiles = np.genfromtxt(pathTele, delimiter=',', dtype='str', encoding='utf-8-sig')
    numTele = len(teleFiles)
    print("Read in teleconncetion indices!")

    inTele = []
    for i in range(numTele):
        file = teleFiles[i]
        path = cwd + '\\Tele\\' + file + '.csv'
        #Read in the teleconnection index data
        inTele.append(np.genfromtxt(path, delimiter=',', skip_header=1))
    
    data = np.stack((inTele[0][:,0], inTele[0][:,1], inTele[0][:,2], inTele[1][:, 2], inTele[2][:,2], inTele[3][:,2], inTele[4][:,2], inTele[5][:,2]), axis=1)

    cellTele = []
    for k in range(numGrids):
        cellTele.append(data)

    #Find start and end row for the specified years
    length = len(cellTele[0])
    startIndex = np.where(cellTele[0][:,0] == startYear)[0].min()
    endIndex = np.where(cellTele[0][:,0] == endYear)[0].max()

    #Trimming down the size of the teleconnection index data to the specified years
    tele = cellTele
    if endIndex < length:
        for i in range(numGrids):
            tele[i] = np.delete(tele[i], slice(endIndex+1, length), axis=0)
    if startIndex > 0:
        for i in range(numGrids):
            tele[i] = np.delete(tele[i], slice(0, startIndex), axis=0)
    
    print("Teleconncetion indices successfully read in for the specified WATFLOOD gird!")

    return tele

def all_data_format_condense(outputNARR, dataGEO, tele, cwd):
    #JG -- These arrays are all loaded atonce in matlab, a feature which does not translate to python
    #    I have split them up into their respective files and loaded them in individually.
    files_to_load = ['geoStatsA.csv', 'geoStatsB.csv', 'isotopeStatsA.csv', 'isotopeStatsB.csv', 'teleStatsA.csv', 'teleStatsB.csv', 'NARRStatsA.csv', 'NARRStatsB.csv']
    geoStats = []
    isotopeStats = []
    teleStats = []
    NARRStats = []
    for file in files_to_load:
        pathFile = cwd + "\\Stats\\" + file
        if 'geo' in file:
            geoStats.append(np.genfromtxt(pathFile, delimiter=',', encoding='utf-8-sig'))
        elif 'iso' in file:
            isotopeStats.append(np.genfromtxt(pathFile, delimiter=',', encoding='utf-8-sig'))
        elif 'tele' in file:
            teleStats.append(np.genfromtxt(pathFile, delimiter=',', encoding='utf-8-sig'))
        elif 'NARR' in file:
            NARRStats.append(np.genfromtxt(pathFile, delimiter=',', encoding='utf-8-sig'))
    dataStats = [geoStats, isotopeStats, teleStats, NARRStats]
    #JG -- These are the same as the matlab code, but I have changed the names to be more pythonic
    #    I have also changed the way the data is stored, instead of cell arrays, it is a list of arrays
    numGrids = len(outputNARR)

    print("Standardizing by Season!")

    for k in range(numGrids):
        #Transform necessary climate variables!
        #At this time, hgt_tropo, pres_tropo, PWAT and apcp all require natural log transformations
        outputNARR[k][:, 11] = np.log(outputNARR[k][:, 11])
        outputNARR[k][:, 12] = np.log(outputNARR[k][:, 12])
        outputNARR[k][:, 15] = np.log(outputNARR[k][:, 15])
        outputNARR[k][:, 16] = np.log(outputNARR[k][:, 16])
    
    #Initializing variables
    outputNARR_DJF = []
    outputNARR_MAM = []
    outputNARR_JJA = []
    outputNARR_SON = []
    tele_DJF = []
    geo_DJF = []
    tele_MAM = []
    geo_MAM = []
    tele_JJA = []
    geo_JJA = []
    tele_SON = []
    geo_SON = []

    #Separate the data into seasonal cells and standardize the data for each season
    #This is done through slicing rather than deleting as was the case in the Matlab code

    #Slicing the arrays into seasonal cells
    for k in range(numGrids):
        #Sorting the Output NARR data into seasonal cells
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 1, outputNARR[k][:, 1] == 2, outputNARR[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 3, outputNARR[k][:, 1] == 4, outputNARR[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 6, outputNARR[k][:, 1] == 7, outputNARR[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 9, outputNARR[k][:, 1] == 10, outputNARR[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])

        #Sorting the teleconnection data into seasonal cells
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 1, tele[k][:, 1] == 2, tele[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 3, tele[k][:, 1] == 4, tele[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 6, tele[k][:, 1] == 7, tele[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 9, tele[k][:, 1] == 10, tele[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])

        #Sorting the geo data into seasonal cells
        #JG -- given the different format of the geo data, I have to do this differently as it does not have a date associated with it
        #   I have to use the index of the outputNARR to sort the geo data
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 1, outputNARR[k][:, 1] == 2, outputNARR[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 3, outputNARR[k][:, 1] == 4, outputNARR[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 6, outputNARR[k][:, 1] == 7, outputNARR[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 9, outputNARR[k][:, 1] == 10, outputNARR[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
    
    #Initializing the standardized seasonal cells
    allCellData_DJF = []
    allCellData_MAM = []
    allCellData_JJA = []
    allCellData_SON = []

    teleSTD_DJF = [np.zeros((tele_DJF[0].shape)) for _ in range(numGrids)]
    teleSTD_MAM = [np.zeros((tele_MAM[0].shape)) for _ in range(numGrids)]
    teleSTD_JJA = [np.zeros((tele_JJA[0].shape)) for _ in range(numGrids)]
    teleSTD_SON = [np.zeros((tele_SON[0].shape)) for _ in range(numGrids)]

    geoSTD_DJF = [np.zeros((geo_DJF[0].shape)) for _ in range(numGrids)]
    geoSTD_MAM = [np.zeros((geo_MAM[0].shape)) for _ in range(numGrids)]
    geoSTD_JJA = [np.zeros((geo_JJA[0].shape)) for _ in range(numGrids)]
    geoSTD_SON = [np.zeros((geo_SON[0].shape)) for _ in range(numGrids)]

    narrSTD_DJF = [np.zeros((outputNARR_DJF[0].shape)) for _ in range(numGrids)]
    narrSTD_MAM = [np.zeros((outputNARR_MAM[0].shape)) for _ in range(numGrids)]
    narrSTD_JJA = [np.zeros((outputNARR_JJA[0].shape)) for _ in range(numGrids)]
    narrSTD_SON = [np.zeros((outputNARR_SON[0].shape)) for _ in range(numGrids)]

    for k in range(numGrids):
        #Standardizing the NARR Data
        colsNARR = outputNARR_DJF[0].shape[1]
        narrSTD_DJF[k][:, 0:3] = outputNARR_DJF[k][:, 0:3]
        narrSTD_MAM[k][:, 0:3] = outputNARR_MAM[k][:, 0:3]
        narrSTD_JJA[k][:, 0:3] = outputNARR_JJA[k][:, 0:3]
        narrSTD_SON[k][:, 0:3] = outputNARR_SON[k][:, 0:3]
        for m in range(3, colsNARR):
            adjM = m-3
            narrSTD_DJF[k][:, m] = (outputNARR_DJF[k][:, m] - NARRStats[0][adjM, 0])/NARRStats[1][adjM, 0]
            narrSTD_MAM[k][:, m] = (outputNARR_MAM[k][:, m] - NARRStats[0][adjM, 1])/NARRStats[1][adjM, 1]
            narrSTD_JJA[k][:, m] = (outputNARR_JJA[k][:, m] - NARRStats[0][adjM, 2])/NARRStats[1][adjM, 2]
            narrSTD_SON[k][:, m] = (outputNARR_SON[k][:, m] - NARRStats[0][adjM, 3])/NARRStats[1][adjM, 3]
        #Standardizing the teleconnection data
        colsTele = tele_DJF[0].shape[1]
        teleSTD_DJF[k][:, 0:2] = tele_DJF[k][:, 0:2]
        teleSTD_MAM[k][:, 0:2] = tele_MAM[k][:, 0:2]
        teleSTD_JJA[k][:, 0:2] = tele_JJA[k][:, 0:2]
        teleSTD_SON[k][:, 0:2] = tele_SON[k][:, 0:2]
        for m in range(2, colsTele):
            adjM = m-2
            teleSTD_DJF[k][:, m] = (tele_DJF[k][:, m] - teleStats[0][adjM, 0])/teleStats[1][adjM, 0]
            teleSTD_MAM[k][:, m] = (tele_MAM[k][:, m] - teleStats[0][adjM, 1])/teleStats[1][adjM, 1]
            teleSTD_JJA[k][:, m] = (tele_JJA[k][:, m] - teleStats[0][adjM, 2])/teleStats[1][adjM, 2]
            teleSTD_SON[k][:, m] = (tele_SON[k][:, m] - teleStats[0][adjM, 3])/teleStats[1][adjM, 3]

        #Standardizing the geo data
        colsGeo = geo_DJF[0].shape[1]
        for m in range(colsGeo):
            geoSTD_DJF[k][:, m] = (geo_DJF[k][:, m] - geoStats[0][m, 0])/geoStats[1][m, 0]
            geoSTD_MAM[k][:, m] = (geo_MAM[k][:, m] - geoStats[0][m, 1])/geoStats[1][m, 1]
            geoSTD_JJA[k][:, m] = (geo_JJA[k][:, m] - geoStats[0][m, 2])/geoStats[1][m, 2]
            geoSTD_SON[k][:, m] = (geo_SON[k][:, m] - geoStats[0][m, 3])/geoStats[1][m, 3]
    
        #Combining the standardized seasonal cells into one cell
        allCellData_DJF.append(np.column_stack((narrSTD_DJF[k][:, 0:3], geoSTD_DJF[k][:, 0:3], narrSTD_DJF[k][:, 3:24],  teleSTD_DJF[k][:, 2:9], geoSTD_DJF[k][:, 3:5])))
        allCellData_MAM.append(np.column_stack((narrSTD_MAM[k][:, 0:3], geoSTD_MAM[k][:, 0:3], narrSTD_MAM[k][:, 3:24],  teleSTD_MAM[k][:, 2:9], geoSTD_MAM[k][:, 3:5])))
        allCellData_JJA.append(np.column_stack((narrSTD_JJA[k][:, 0:3], geoSTD_JJA[k][:, 0:3], narrSTD_JJA[k][:, 3:24],  teleSTD_JJA[k][:, 2:9], geoSTD_JJA[k][:, 3:5])))
        allCellData_SON.append(np.column_stack((narrSTD_SON[k][:, 0:3], geoSTD_SON[k][:, 0:3], narrSTD_SON[k][:, 3:24],  teleSTD_SON[k][:, 2:9], geoSTD_SON[k][:, 3:5])))

    #Combining the cell data into one array for each season
    length, width = allCellData_DJF[0].shape
    allData_DJF = np.zeros((length*numGrids, width))
    allData_MAM = np.zeros((length*numGrids, width))
    allData_JJA = np.zeros((length*numGrids, width))
    allData_SON = np.zeros((length*numGrids, width))

    #Stacking the data into one array
    for k in range(numGrids):
        allData_DJF[length*k:length*(k+1), :] = allCellData_DJF[k]
        allData_MAM[length*k:length*(k+1), :] = allCellData_MAM[k]
        allData_JJA[length*k:length*(k+1), :] = allCellData_JJA[k]
        allData_SON[length*k:length*(k+1), :] = allCellData_SON[k]
    print("NARR, geographic, and teleconnection data standardized by season!")
    
    #Sort Data and put into corresponding KPN zone
    #DJF
    sortKPN_DJF = allData_DJF[allData_DJF[:, 32].argsort(kind='mergesort')]
    insertKPN_DJF = np.array([allData_DJF[:, 32].min()])
    colKPN_DJF = np.append(insertKPN_DJF, sortKPN_DJF[:, 32])
    lengthKPN = len(colKPN_DJF)
    diffKPN = sortKPN_DJF[:, 32]-colKPN_DJF[0:lengthKPN-1]
    indexKPN_DJF = np.nonzero(diffKPN) 
    numPts = np.array(len(sortKPN_DJF))
    indexKPN_DJF = np.append(indexKPN_DJF, numPts)
    numIndex = len(indexKPN_DJF)
    allDataKPN_DJF = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i != numIndex-1:
            if np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[0:indexKPN_DJF[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
    
    #MAM
    sortKPN_MAM = allData_MAM[allData_MAM[:, 32].argsort(kind='mergesort')]
    insertKPN_MAM = np.array([allData_MAM[:, 32].min()])
    colKPN_MAM = np.append(insertKPN_MAM, sortKPN_MAM[:, 32])
    lengthKPN = len(colKPN_MAM)
    diffKPN = sortKPN_MAM[:, 32]-colKPN_MAM[0:lengthKPN-1]
    indexKPN_MAM = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_MAM))
    indexKPN_MAM = np.append(indexKPN_MAM, numPts)
    numIndex = len(indexKPN_MAM)
    allDataKPN_MAM = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 35:
                allDataKPN_MAM[0] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 42:
                allDataKPN_MAM[1] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 43:
                allDataKPN_MAM[2] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 47:
                allDataKPN_MAM[3] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 62:
                allDataKPN_MAM[4] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[0:indexKPN_MAM[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
    
    #JJA
    sortKPN_JJA = allData_JJA[allData_JJA[:, 32].argsort(kind='mergesort')]
    insertKPN_JJA = np.array([allData_JJA[:, 32].min()])
    colKPN_JJA = np.append(insertKPN_JJA, sortKPN_JJA[:, 32])
    lengthKPN = len(colKPN_JJA)
    diffKPN = sortKPN_JJA[:, 32]-colKPN_JJA[0:lengthKPN-1]
    indexKPN_JJA = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_JJA))
    indexKPN_JJA = np.append(indexKPN_JJA, numPts)
    numIndex = len(indexKPN_JJA)
    allDataKPN_JJA = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 35:
                allDataKPN_JJA[0] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 42:
                allDataKPN_JJA[1] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 43:
                allDataKPN_JJA[2] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 47:
                allDataKPN_JJA[3] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 62:
                allDataKPN_JJA[4] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[0:indexKPN_JJA[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
    
    #SON
    sortKPN_SON = allData_SON[allData_SON[:, 32].argsort(kind='mergesort')]
    insertKPN_SON = np.array([allData_SON[:, 32].min()])
    colKPN_SON = np.append(insertKPN_SON, sortKPN_SON[:, 32])
    lengthKPN = len(colKPN_SON)
    diffKPN = sortKPN_SON[:, 32]-colKPN_SON[0:lengthKPN-1]
    indexKPN_SON = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_SON))
    indexKPN_SON = np.append(indexKPN_SON, numPts)
    numIndex = len(indexKPN_SON)
    allDataKPN_SON = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 35:
                allDataKPN_SON[0] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 42:
                allDataKPN_SON[1] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 43:
                allDataKPN_SON[2] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 47:
                allDataKPN_SON[3] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 62:
                allDataKPN_SON[4] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[0:indexKPN_SON[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
    
    #Combing all that data into one list
    dataAllKPN_seas = [allDataKPN_DJF, allDataKPN_MAM, allDataKPN_JJA, allDataKPN_SON]
    
    print("Data has been standardized and condensed into seasons!")
    return dataAllKPN_seas, dataStats

def simulate_Kpn(dataAllKPN_seas, dataStats, cwd):
    #Unpacking datastats
    geoStats = dataStats[0]
    isotopeStats = dataStats[1]
    teleStats = dataStats[2]
    NARRStats = dataStats[3]
    #JG -- Due to the fact that the linear models are presaved in a .mat file in the matlab version I needed a 
    # different solution for this. I created a json File that contains the coeffecients, intercepts, and the affected 
    # rows for each model. THis is used to then calculate the prediction. It is not the same as the OG matlab code unfortunatel
    # and vary from slightly different to very different. Once I have a better solution I will update this.
   
    with open(cwd + r"\ModelData\LinearRegression.json") as f:
        modelKPN_seas = json.load(f)
    

    print("Simulate monthly 18Oppt and prediction intervals for the KPN regionalization.")
    question = "\nWould you like to account for 18Oppt input uncertainty by calculating prediction intervals?\nNOTE: this is a very time consuming, computationally heavy process. Y/N: "
    checkPI = input(question).lower()
    
    if checkPI == "y":
        binaryPI = 1
    else:
        binaryPI = 0
    
    #Read in CELL 'data', and then extract the parameter data into annual cells
    # (one cell for each parameter/variable, 35 in total)
    numSeas = len(dataAllKPN_seas)
    numKPN = len(dataAllKPN_seas[0])
    #Initializing
    allData_Stack = [[] for i in range(numSeas)]
    kpnCol = [[] for i in range(numSeas)]
    kpnDiff = [[] for i in range(numSeas)]
    indexKpn = [[] for i in range(numSeas)]

    for i in range(numSeas):
        for j in range(numKPN):
            if len(dataAllKPN_seas[i][j]) > 0:
                if len(allData_Stack[i]) == 0:
                    allData_Stack[i] = dataAllKPN_seas[i][j]
                else:
                    allData_Stack[i] = np.vstack((allData_Stack[i], dataAllKPN_seas[i][j]))
            else:
                continue
        
        #Separate the data into grids (i.e. each cell index is a new grid)
        allData_Stack[i] = allData_Stack[i][allData_Stack[i][:, 0].argsort(kind = 'mergesort')]
        insertKPN = np.array(allData_Stack[i][0,0])
        kpnCol[i] = np.append(insertKPN, allData_Stack[i][:, 0])
        kpnLength = len(kpnCol[i])
        kpnDiff[i] = allData_Stack[i][:,0] - kpnCol[i][:kpnLength-1]
        indexKpn[i]=np.argwhere(kpnDiff[i]) #Ran into issue here
        numPts = np.array(len(allData_Stack[i]))
        indexKpn[i] = np.append(indexKpn[i], numPts)
        numIndex = len(indexKpn[i])
    
    #Initializing the grids for the variables
    allKPN_Grids = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    month = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    year = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    lat = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    lon = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    alt = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    apcp = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    cape = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    cdcon = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    cdlyr = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    evap = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    hcdc = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    hbpl = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    mcdc = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    prwtr = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    rhum_2m = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    uwnd_10m = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    vwnd_10m = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    wcconv = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    wcvflx = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    amo = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    ao = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    nao = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    pdo = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    pna = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    soi = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    kpnZone = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    kpnID = np.zeros((4, numIndex))
    sinVar = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    cosVar = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    xKPN = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    xKPN2 = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    pi = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    stackPI = [[] for i in range(numIndex)]

    print("Preparing Model Inputs")

    for i in range(numSeas):
        for m in range(numIndex):
            if m == 0:
                if m != numIndex-1:
                    allKPN_Grids[i][m] = allData_Stack[i][:indexKpn[i][m], :]
                else:
                    allKPN_Grids[i][m] = allData_Stack[i][:, :]
            else:
                if m != numIndex-1:
                    allKPN_Grids[i][m] = allData_Stack[i][indexKpn[i][m-1]:indexKpn[i][m], :]
                else:
                    allKPN_Grids[i][m] = allData_Stack[i][indexKpn[i][m-1]:, :]
            
            month[i][m] = allKPN_Grids[i][m][:, 1]
            year[i][m] = allKPN_Grids[i][m][:, 2]
            lat[i][m] = allKPN_Grids[i][m][:, 3]
            lon[i][m] = allKPN_Grids[i][m][:, 4]
            alt[i][m] = allKPN_Grids[i][m][:, 5]
            #acpcp is unused ???
            #air2m is unused ???
            apcp[i][m] = allKPN_Grids[i][m][:, 8]
            cape[i][m] = allKPN_Grids[i][m][:, 9]
            cdcon[i][m] = allKPN_Grids[i][m][:, 10]
            cdlyr[i][m] = allKPN_Grids[i][m][:, 11]
            evap[i][m] = allKPN_Grids[i][m][:, 12]
            hcdc[i][m] = allKPN_Grids[i][m][:, 13]
            #hgt_tropo unused ???
            hbpl[i][m] = allKPN_Grids[i][m][:, 15]
            #lcdc unused ???
            mcdc[i][m] = allKPN_Grids[i][m][:, 17]
            #pres_topo unused ???
            prwtr[i][m] = allKPN_Grids[i][m][:, 19]
            rhum_2m[i][m] = allKPN_Grids[i][m][:, 20]
            #tcdc unused ???
            uwnd_10m[i][m] = allKPN_Grids[i][m][:, 22]
            vwnd_10m[i][m] = allKPN_Grids[i][m][:, 23]
            wcconv[i][m] = allKPN_Grids[i][m][:, 24]
            wcvflx[i][m] = allKPN_Grids[i][m][:, 25]
            amo[i][m] = allKPN_Grids[i][m][:, 26]
            ao[i][m] = allKPN_Grids[i][m][:, 27]
            nao[i][m] = allKPN_Grids[i][m][:, 28]
            pdo[i][m] = allKPN_Grids[i][m][:, 29]
            pna[i][m] = allKPN_Grids[i][m][:, 30]
            soi[i][m] = allKPN_Grids[i][m][:, 31]
            kpnZone[i][m] = allKPN_Grids[i][m][:, 32]
            sinVar[i][m] = np.sin(2*np.pi*(allKPN_Grids[i][m][:, 1]-1)/12)
            cosVar[i][m] = np.cos(2*np.pi*(allKPN_Grids[i][m][:, 1]-1)/12)
    
    #Before starting the simulation, must separate the variables that the KPN model
    # uses from the dataALL_cell

    for m in range(numIndex):
        rows0 = len(allKPN_Grids[0][m])
        rows1 = len(allKPN_Grids[1][m])
        rows2 = len(allKPN_Grids[2][m])
        rows3 = len(allKPN_Grids[3][m])
        if kpnZone[0][m][0] == 35:
            xKPN2[0][m] = np.transpose(np.vstack((np.ones(rows0), lat[0][m][:], alt[0][m][:])))
            xKPN[0][m] = np.transpose(np.vstack((sinVar[0][m][:], cosVar[0][m][:], apcp[0][m][:],\
                                    cape[0][m][:], cdcon[0][m][:], cdlyr[0][m][:], evap[0][m][:],\
                                    hcdc[0][m][:], hbpl[0][m][:], mcdc[0][m][:], prwtr[0][m][:],\
                                    rhum_2m[0][m][:], uwnd_10m[0][m][:], vwnd_10m[0][m][:],\
                                    wcconv[0][m][:], wcvflx[0][m][:], amo[0][m][:],\
                                    ao[0][m][:], nao[0][m][:], pdo[0][m][:],\
                                    pna[0][m][:], soi[0][m][:], lat[0][m][:], lon[0][m][:], alt[0][m][:])))
            xKPN2[1][m] = np.transpose(np.vstack((np.ones(rows1), sinVar[1][m][:], cosVar[1][m][:], lon[1][m][:])))
            xKPN[1][m] = np.transpose(np.vstack((sinVar[1][m][:], cosVar[1][m][:], lon[1][m][:])))
            xKPN2[2][m] = np.transpose(np.vstack((np.ones(rows2), hbpl[2][m][:], mcdc[2][m][:], wcvflx[2][m][:])))
            xKPN[2][m] = np.transpose(np.vstack((sinVar[2][m][:], cosVar[2][m][:], apcp[2][m][:],\
                                    cape[2][m][:], cdcon[2][m][:], cdlyr[2][m][:], evap[2][m][:],\
                                    hcdc[2][m][:], hbpl[2][m][:], mcdc[2][m][:], prwtr[2][m][:],\
                                    rhum_2m[2][m][:], uwnd_10m[2][m][:], vwnd_10m[2][m][:],\
                                    wcconv[2][m][:], wcvflx[2][m][:], amo[2][m][:],\
                                    ao[2][m][:], nao[2][m][:], pdo[2][m][:],\
                                    pna[2][m][:], soi[2][m][:], lat[2][m][:], lon[2][m][:], alt[2][m][:])))
            xKPN2[3][m] = np.transpose(np.vstack((np.ones(rows3), prwtr[3][m][:], wcvflx[3][m][:], lat[3][m][:], alt[3][m][:])))
            xKPN[3][m] = np.transpose(np.vstack((sinVar[3][m][:], cosVar[3][m][:], apcp[3][m][:], cape[3][m][:],\
                                    cdcon[3][m][:], cdlyr[3][m][:], evap[3][m][:], hcdc[3][m][:],\
                                    hbpl[3][m][:], mcdc[3][m][:], prwtr[3][m][:], rhum_2m[3][m][:],\
                                    uwnd_10m[3][m][:], vwnd_10m[3][m][:], wcconv[3][m][:],\
                                    wcvflx[3][m][:], amo[3][m][:], ao[3][m][:], nao[3][m][:],\
                                    pdo[3][m][:], pna[3][m][:], soi[3][m][:], lat[3][m][:],\
                                    lon[3][m][:], alt[3][m][:])))
            kpnID[0:4, m] = 1
        elif kpnZone[0][m][0] == 42:
            xKPN2[0][m] = np.transpose(np.vstack((np.ones(rows0), sinVar[0][m][:], cosVar[0][m][:], cdlyr[0][m][:],\
                                    hbpl[0][m][:], prwtr[0][m][:], lon[0][m][:])))
            xKPN[0][m] = np.transpose(np.vstack((sinVar[0][m][:], cosVar[0][m][:], cdlyr[0][m][:], hbpl[0][m][:],\
                                    prwtr[0][m][:], lon[0][m][:])))
            xKPN2[1][m] = np.transpose(np.vstack((np.ones(rows1), mcdc[1][m][:], prwtr[1][m][:], rhum_2m[1][m][:],\
                                    lat[1][m][:], lon[1][m][:], (lat[1][m][:]*lon[1][m][:]))))
            xKPN[1][m] = np.transpose(np.vstack((sinVar[1][m][:], cosVar[1][m][:], apcp[1][m][:], cape[1][m][:],\
                                    cdcon[1][m][:], cdlyr[1][m][:], evap[1][m][:], hcdc[1][m][:],\
                                    hbpl[1][m][:], mcdc[1][m][:], prwtr[1][m][:], rhum_2m[1][m][:],\
                                    uwnd_10m[1][m][:], vwnd_10m[1][m][:], wcconv[1][m][:],\
                                    wcvflx[1][m][:], amo[1][m][:], ao[1][m][:], nao[1][m][:],\
                                    pdo[1][m][:], pna[1][m][:], soi[1][m][:], lat[1][m][:],\
                                    lon[1][m][:], alt[1][m][:])))
            xKPN2[2][m] = np.transpose(np.vstack((np.ones(rows2), mcdc[2][m][:], prwtr[2][m][:], uwnd_10m[2][m][:],\
                                    vwnd_10m[2][m][:], lat[2][m][:], lon[2][m][:], (mcdc[2][m][:]*uwnd_10m[2][m][:]),\
                                    (mcdc[2][m][:]*lon[2][m][:]), lat[2][m][:]*lon[2][m][:])))
            xKPN[2][m] = np.transpose(np.vstack((sinVar[2][m][:], cosVar[2][m][:], apcp[2][m][:], cape[2][m][:],\
                                    cdcon[2][m][:], cdlyr[2][m][:], evap[2][m][:], hcdc[2][m][:],\
                                    hbpl[2][m][:], mcdc[2][m][:], prwtr[2][m][:], rhum_2m[2][m][:],\
                                    uwnd_10m[2][m][:], vwnd_10m[2][m][:], wcconv[2][m][:],\
                                    wcvflx[2][m][:], amo[2][m][:], ao[2][m][:], nao[2][m][:],\
                                    pdo[2][m][:], pna[2][m][:], soi[2][m][:], lat[2][m][:],\
                                    lon[2][m][:], alt[2][m][:])))
            xKPN2[3][m] = np.transpose(np.vstack((np.ones(rows3), mcdc[3][m][:], prwtr[3][m][:], rhum_2m[3][m][:],\
                                    pdo[3][m][:], lat[3][m][:], lon[3][m][:], (cape[3][m][:]*prwtr[3][m][:]),\
                                    (cape[3][m][:]*lon[3][m][:]), (mcdc[3][m][:]*lat[3][m][:]),\
                                    (prwtr[3][m][:]*rhum_2m[3][m][:]), (prwtr[3][m][:])**2)))
            xKPN[3][m] = np.transpose(np.vstack((mcdc[3][m][:], prwtr[3][m][:], rhum_2m[3][m][:], pdo[3][m][:],\
                                    lat[3][m][:], lon[3][m][:], (cape[3][m][:]*prwtr[3][m][:]),\
                                    (cape[3][m][:]*lon[3][m][:]), (mcdc[3][m][:]*lat[3][m][:]),\
                                    (prwtr[3][m][:]*rhum_2m[3][m][:]), (prwtr[3][m][:])**2)))
            kpnID[0:4, m] = 2
        elif kpnZone[0][m][0] == 43:
            xKPN2[0][m] = np.transpose(np.vstack((np.ones(rows0), evap[0][m][:], prwtr[0][m][:], nao[0][m][:], pdo[0][m][:],\
                                    lon[0][m][:], (lon[0][m][:])**2)))
            xKPN[0][m] = np.transpose(np.vstack((sinVar[0][m][:], cosVar[0][m][:], apcp[0][m][:], cape[0][m][:],\
                                    cdcon[0][m][:], cdlyr[0][m][:], evap[0][m][:], hcdc[0][m][:],\
                                    hbpl[0][m][:], mcdc[0][m][:], prwtr[0][m][:], rhum_2m[0][m][:],\
                                    uwnd_10m[0][m][:], vwnd_10m[0][m][:], wcconv[0][m][:], wcvflx[0][m][:],\
                                    amo[0][m][:], ao[0][m][:], nao[0][m][:], pdo[0][m][:], pna[0][m][:],\
                                    soi[0][m][:], lat[0][m][:], lon[0][m][:], alt[0][m][:])))
            xKPN2[1][m] = np.transpose(np.vstack((np.ones(rows1), mcdc[1][m][:], prwtr[1][m][:], pna[1][m][:],\
                                    lat[1][m][:], alt[1][m][:], (evap[1][m][:]*alt[1][m][:]),\
                                    (prwtr[1][m][:]*alt[1][m][:]), (lat[1][m][:]*alt[1][m][:]), (alt[1][m][:])**2)))
            xKPN[1][m] = np.transpose(np.vstack((mcdc[1][m][:], prwtr[1][m][:], pna[1][m][:], lat[1][m][:],\
                                    alt[1][m][:], (evap[1][m][:]*alt[1][m][:]), (prwtr[1][m][:]*alt[1][m][:]),\
                                    (lat[1][m][:]*alt[1][m][:]), (alt[1][m][:])**2)))
            xKPN2[2][m] = np.transpose(np.vstack((np.ones(rows2), apcp[2][m][:], vwnd_10m[2][m][:], lat[2][m][:], lon[2][m][:])))
            xKPN[2][m] = np.transpose(np.vstack((sinVar[2][m][:], cosVar[2][m][:], apcp[2][m][:], cape[2][m][:],\
                                    cdcon[2][m][:], cdlyr[2][m][:], evap[2][m][:], hcdc[2][m][:],\
                                    hbpl[2][m][:], mcdc[2][m][:], prwtr[2][m][:], rhum_2m[2][m][:],\
                                    uwnd_10m[2][m][:], vwnd_10m[2][m][:], wcconv[2][m][:], wcvflx[2][m][:],\
                                    amo[2][m][:], ao[2][m][:], nao[2][m][:], pdo[2][m][:], pna[2][m][:],\
                                    soi[2][m][:], lat[2][m][:], lon[2][m][:], alt[2][m][:])))
            xKPN2[3][m] = np.transpose(np.vstack((np.ones(rows3), mcdc[3][m][:], prwtr[3][m][:], uwnd_10m[3][m][:],\
                                    amo[3][m][:], lat[3][m][:], (mcdc[3][m][:])**2)))
            xKPN[3][m] = np.transpose(np.vstack((sinVar[3][m][:], cosVar[3][m][:], apcp[3][m][:], cape[3][m][:],\
                                    cdcon[3][m][:], cdlyr[3][m][:], evap[3][m][:], hcdc[3][m][:],\
                                    hbpl[3][m][:], mcdc[3][m][:], prwtr[3][m][:], rhum_2m[3][m][:],\
                                    uwnd_10m[3][m][:], vwnd_10m[3][m][:], wcconv[3][m][:], wcvflx[3][m][:],\
                                    amo[3][m][:], ao[3][m][:], nao[3][m][:], pdo[3][m][:], pna[3][m][:],\
                                    soi[3][m][:], lat[3][m][:], lon[3][m][:], alt[3][m][:])))
            kpnID[0:4, m] = 3
        elif kpnZone[0][m][0] == 47:
            xKPN2[0][m] = np.transpose(np.vstack((np.ones(rows0), sinVar[0][m][:], cosVar[0][m][:])))
            xKPN[0][m] = np.transpose(np.vstack((sinVar[0][m][:], cosVar[0][m][:])))
            xKPN2[1][m] = np.transpose(np.vstack((np.ones(rows1), sinVar[1][m][:], cosVar[1][m][:])))
            xKPN[1][m] = np.transpose(np.vstack((sinVar[1][m][:], cosVar[1][m][:])))
            xKPN2[2][m] = np.transpose(np.vstack((np.ones(rows2), sinVar[2][m][:], cosVar[2][m][:])))
            xKPN[2][m] = np.transpose(np.vstack((sinVar[2][m][:], cosVar[2][m][:])))
            xKPN2[3][m] = np.transpose(np.vstack((np.ones(rows3), sinVar[3][m][:], cosVar[3][m][:])))
            xKPN[3][m] = np.transpose(np.vstack((sinVar[3][m][:], cosVar[3][m][:])))
            kpnID[0:3+1, m] = 4
        elif kpnZone[0][m][0] == 62:
            xKPN2[0][m] = np.transpose(np.vstack((np.ones(rows0), sinVar[0][m][:], cosVar[0][m][:], mcdc[0][m][:], prwtr[0][m][:], lon[0][m][:])))
            xKPN[0][m] = np.transpose(np.vstack((sinVar[0][m][:], cosVar[0][m][:], mcdc[0][m][:], prwtr[0][m][:], lon[0][m][:])))
            xKPN2[1][m] = np.transpose(np.vstack((np.ones(rows1), sinVar[1][m][:], cosVar[1][m][:], prwtr[1][m][:], ao[1][m][:])))
            xKPN[1][m] = np.transpose(np.vstack((sinVar[1][m][:], cosVar[1][m][:], prwtr[1][m][:], ao[1][m][:])))
            xKPN2[2][m] = np.transpose(np.vstack((np.ones(rows2), prwtr[2][m][:], vwnd_10m[2][m][:])))
            xKPN[2][m] = np.transpose(np.vstack((sinVar[2][m][:], cosVar[2][m][:], apcp[2][m][:], cape[2][m][:], cdcon[2][m][:],\
                                   cdlyr[2][m][:], evap[2][m][:], hcdc[2][m][:], hbpl[2][m][:], mcdc[2][m][:],\
                                    prwtr[2][m][:], rhum_2m[2][m][:], uwnd_10m[2][m][:], vwnd_10m[2][m][:],\
                                    wcconv[2][m][:], wcvflx[2][m][:], amo[2][m][:], ao[2][m][:], nao[2][m][:],\
                                    pdo[2][m][:], pna[2][m][:], soi[2][m][:], lat[2][m][:], lon[2][m][:], alt[2][m][:])))
            xKPN2[3][m] = np.transpose(np.vstack((np.ones(rows3), mcdc[3][m][:], prwtr[3][m][:], lat[3][m][:])))
            xKPN[3][m] = np.transpose(np.vstack((mcdc[3][m][:], prwtr[3][m][:], lat[3][m][:])))
            kpnID[0:4, m] = 5
    print("Simulating monthly 18Oppt!")

    ypred_std_KPN = [[[] for j  in range(numIndex)] for i in range(numSeas)]
    if checkPI == 'y': # Currently skipping this section will develop at a later point
        # numR is the number of iterations (typically 1000 is enough) and alpha is the significance level
        # these can be changed if required
        numR = 1000
        alpha = 0.05
        indexUp = numR * (1-alpha)
        indexDown = numR * alpha

        for season in range(numSeas):
            for gridNum in range(numIndex):
                numX, numVar = np.shape(xKPN[season][gridNum])
                model = modelKPN_seas[season][int(kpnID[season][gridNum]-1)]
                print("Not finished development yet")
    else:
        for season in range(numSeas):
            for gridNum in range(numIndex):
                numX = np.shape(xKPN[season][gridNum])[0]
                modelNum =  "LM" + str(season*5+int(kpnID[season][gridNum]))
                model = modelKPN_seas[modelNum]
                rowArray = np.zeros((numX, 1))
                for row in range(numX):
                    xValue = 0
                    for col, coeff in model.items():
                        if col == "Intercept":
                            xValue += coeff
                        elif len(col.split(',')) == 2 and col.split(',')[0] == col.split(',')[1]:
                            xValue += coeff * xKPN[season][gridNum][row, int(col.split(',')[0])]
                        elif len(col.split(',')) == 2 and col.split(',')[0] != col.split(',')[1]:
                            xValue += coeff * xKPN[season][gridNum][row, int(col.split(',')[0])] * xKPN[season][gridNum][row, int(col.split(',')[1])]
                        else:
                            xValue += coeff * xKPN[season][gridNum][row, int(col)]
                    rowArray[row] = xValue
                ypred_std_KPN[season][gridNum] = rowArray

                colOne = month[season][gridNum]
                colTwo = year[season][gridNum]
                colThree = np.multiply(ypred_std_KPN[season][gridNum], isotopeStats[1][0, season]) + isotopeStats[0][0, season]
                colFour = np.full(colOne.shape[0], gridNum+1)
                
                pi[season][gridNum] = np.column_stack((colOne, colTwo, colThree[:, 0], colFour))
    
    #Stacking the PI's into a single array then sorting based on month and year
    for m in range(numIndex):
        stackPI[m] = np.vstack((pi[0][m], pi[1][m], pi[2][m], pi[3][m]))
        stackPI[m] = stackPI[m][stackPI[m][:, 1].argsort(kind = 'mergesort')]
        stackPI[m] = stackPI[m][stackPI[m][:, 0].argsort(kind = 'mergesort')]

    print("Time series 18Oppt for the KPN regionalization has been simulated. YAY!")
    return stackPI, binaryPI

def write_Kpn_WATFLOOD(path, basinName, stackPI, binaryPI):
########################################################################
#
#    Comments: This program reads in the PI simulations,
#    re-formats them into a cell (each cell index represents a monthly gridded
#    18Oppt for the basin, with num_month indices). Then the program writes
#    the gridded 18Oppt time series to the r2c file format (including the
#    header) and outputs it as yyyymmdd_drn.r2c.
#
#    TH: added monthly/yearly output option for different WATFLOOD events
########################################################################


    # Ask user if they would like to output the data in monthly or yearly format
    userDecision = input("Would you like yearly or monthly output? Y/M: ").lower()
    
    # Read through shd file to determine some base information
    headerInfo = {}
    # Read in the header from the SHD file 
    with open(path + f"\\basin\\{basinName}_shd.r2c", 'r') as f:
        for line in f:
            if line.startswith(":SourceFileName"):
                headerInfo["SourceFileName"] = line.split()[1]
            elif line.startswith(":Projection"):
                headerInfo["Projection"] = line.split()[1]
            elif line.startswith(":Ellipsoid"):
                headerInfo["Ellipsoid"] = line.split()[1]
            elif line.startswith(":xOrigin"):
                headerInfo["xOrigin"] = line.split()[1]
            elif line.startswith(":yOrigin"):
                headerInfo["yOrigin"] = line.split()[1]
            elif line.startswith(":xCount"):
                headerInfo["xCount"] = line.split()[1]
            elif line.startswith(":yCount"):
                headerInfo["yCount"] = line.split()[1]
            elif line.startswith(":xDelta"):
                headerInfo["xDelta"] = line.split()[1]
            elif line.startswith(":yDelta"):
                headerInfo["yDelta"] = line.split()[1]
    
    # Add other information to the header
    headerInfo["CreationDate"] = datetime.now().strftime("%Y/%m/%d")
    headerInfo["WrittenBy"] = "isoP.py"
    headerInfo["DataType"] = "2D Rect Cell"
    headerInfo["Name"] = "18Oppt_KPN_mean"
    headerInfo["Units"] = "permille"
    headerInfo["AtributeName"] = "deltar"
    headerInfo["UnitConversion"] = "1.0000"
    headerInfo["SourceFileName"] = "N/A"

    # Creating the header within a single string
    header = f"""########################################
:FileType r2c  ASCII  EnSim 1.0
#
# DataType               {headerInfo["DataType"]}
#
:Application             EnSimHydrologic
:Version                 1.0
:WrittenBy          {headerInfo["WrittenBy"]}
:CreationDate         {headerInfo["CreationDate"]}
#
#---------------------------------------
#
:Name          {headerInfo["Name"]}
#
:Projection         {headerInfo["Projection"]}
:Ellipsoid         {headerInfo["Ellipsoid"]}
#
:xOrigin         {headerInfo["xOrigin"]}
:yOrigin         {headerInfo["yOrigin"]}
#
:SourceFile                   {headerInfo["SourceFileName"]}
#
:AttributeName 1    {headerInfo["AtributeName"]}
:AttributeUnits     {headerInfo["Units"]}
#
:xCount         {headerInfo["xCount"]}
:yCount         {headerInfo["yCount"]}
:xDelta         {headerInfo["xDelta"]}
:yDelta         {headerInfo["yDelta"]}
#
:UnitConversion               {headerInfo["UnitConversion"]}
#
:endHeader"""

    
    # Change format of output from a cell for each grid to a cell for each month in entire grid
    numMonths = len(stackPI[0])
    numGrids = len(stackPI)
    meanO18_allGrids = np.zeros((numMonths, numGrids)) # Initialize the array to hold the data
    meanO18_Gridded = [np.zeros((int(headerInfo["yCount"]), int(headerInfo["xCount"]))) for i in range(numMonths)]

    for i in range(numGrids):
        if binaryPI == 1:
            # This will be done once I have added this function
            pass
        else:
            meanO18_allGrids[:, i] = stackPI[i][:, 2]

    for m in range(numMonths):
        if binaryPI == 1:
            # This will be done once I have added this function
            pass
        else:
            # Grab the month from meanO18_allGrids
            meanO18_month = meanO18_allGrids[m, :]
            yCount_array = range(0,numGrids+1, int(headerInfo["yCount"])) # CHECK TO SEE IF INDEXING IS CORRECT

        for j in range(int(headerInfo['xCount'])):
            if binaryPI == 1:
                # This will be done once I have added this function
                pass
            else:
                meanO18_Gridded[m][:,j] = meanO18_month[yCount_array[j]:yCount_array[j+1]]

    # Extract the years from the stackPI
    years = set(stackPI[0][:, 1].astype(int))

    # Now to write the data into separate files if they chose a yearly output #WILL NOT DO STATS ONE YET
    if userDecision == 'y':
        for i, year in enumerate(years):
            with open(path + f"\\isoP\\{year}01_drnTEST.r2c", 'w') as file:
                file.write(header)
                # Cycle through the months and write the data
                for month in range(1,13):
                    monthData = np.flipud(meanO18_Gridded[i*12 + month - 1])
                    file.write(f"\n:Frame        {month}        {month}    \"{year}/{month}/1 1:00:00.000\"\n")
                    for row in monthData:
                        file.write("  ".join(['{:.2f}'.format(item) for item in row]) + "\n")
                    file.write(":EndFrame")
    elif userDecision == 'm':
        for i, year in enumerate(years):
            for month in range(1,13):
                with open(path + f"\\isoP\\{year}{str(month).zfill(2)}01_drnTEST.r2c", 'w') as file:
                    file.write(header)
                    monthData = np.flipud(meanO18_Gridded[i*12 + month - 1])
                    
                    # Monthly data for the first day of the month at 1:00:00.000
                    file.write(f"\n:Frame        {month}        {month}    \"{year}/{month}/1 1:00:00.000\"\n")
                    for row in monthData:
                        file.write("  ".join(['{:.2f}'.format(item) for item in row]) + "\n")
                    file.write(":EndFrame")
                    
                    # Monthly data for the last day of the month at 24:00:00.000
                    last_day = datetime(year, month, 1).replace(day=1) - timedelta(days=1)
                    file.write(f"\n:Frame        {month}        {month}    \"{year}/{month}/{last_day.day} 24:00:00.000\"\n")
                    for row in monthData:
                        file.write("  ".join(['{:.2f}'.format(item) for item in row]) + "\n")
                    file.write(":EndFrame")
    else:
        print("Invalid input, please put either 'Y' or 'M'")

def write_Kpn_coords(stackPI, path, basinName):
    WFcoords = np.genfromtxt(path + f"\\isoP\\{basinName}_coords.csv", delimiter = ',')

    # Intialize the output file
    outputFile = pd.DataFrame(columns=["Month", "Year", "Latitude", "Longitude", "O18"])
    numMonths = len(stackPI[0])
    for grid in range(len(stackPI)):
        lat = WFcoords[grid][0]
        lon = WFcoords[grid][1]

        gridOutput = pd.DataFrame({"Month": stackPI[grid][:, 0], 
                                    "Year": stackPI[grid][:, 1], 
                                    "Latitude": np.full(numMonths, lat),
                                    "Longitude": np.full(numMonths, lon),
                                    "O18": stackPI[grid][:, 2]})
        outputFile = pd.concat([outputFile, gridOutput],  ignore_index=True)
    outputFile.to_csv(path + f"\\isoP\\{basinName}_O18.csv", index = False)

def download_NARR_data():
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
        print(str(round(count/len(varFile)*100, 2)) + "% complete!")
        count += 1

def isoP_WATFLOOD(cwd, userProfile = None):
    if userProfile == None:
        userInfo = {}
        path = input("\nPlease enter the full path to the parent folder containing the isoP folder:\n")
        userInfo['basinPath'] = path
        basinName = input("\nPlease enter the name of the basin: ")
        userInfo['basinName'] = basinName
        startYear = int(input("\nPlease enter the start year wanted: "))
        userInfo['startYear'] = startYear
        endYear = int(input("\nPlease enter the end year wanted: "))
        userInfo['endYear'] = endYear
        writeUser_Profile(userInfo)
        print("\n")

    else:
        path = userProfile['basinPath']
        basinName = userProfile['basinName']
        startYear = int(userProfile['startYear'])
        endYear = int(userProfile['endYear'])
        print("\n")

    WFcoords = readSHD_File(path, basinName)
    output, pathNARR = extract_NARR_timeseries(cwd, path, basinName)
    outputNARR = NARR_format_timeseries_basin(output, pathNARR, startYear, endYear)
    dataGEO = extract_GIS_info(startYear, endYear, WFcoords, cwd)
    tele = extract_tele_timeseries_basin(WFcoords, cwd, startYear, endYear)
    dataALLKPN_seas, dataStats = all_data_format_condense(outputNARR, dataGEO, tele, cwd)
    stackPI, binaryPI = simulate_Kpn(dataALLKPN_seas, dataStats, cwd)
    write_Kpn_WATFLOOD(path, basinName, stackPI, binaryPI)
    print("isoP has been completed!")

def isoP_coords(cwd, userProfile = None):
    if userProfile == None:
        userInfo = {}
        path = input("\nPlease enter the full path to the parent folder containing the isoP folder:\n")
        userInfo['basinPath'] = path
        basinName = input("\nPlease enter the name of the basin: ")
        userInfo['basinName'] = basinName
        startYear = int(input("\nPlease enter the start year wanted: "))
        userInfo['startYear'] = startYear
        endYear = int(input("\nPlease enter the end year wanted: "))
        userInfo['endYear'] = endYear
        writeUser_Profile(userInfo)
        print("\n")

    else:
        path = userProfile['basinPath']
        basinName = userProfile['basinName']
        startYear = int(userProfile['startYear'])
        endYear = int(userProfile['endYear'])
        print("\n")   

    # Create the path to the coordinates file
    pathCoords = path + f"\\isoP\\{basinName}_coords.csv"

    coords = np.genfromtxt(pathCoords, delimiter = ',')
    output, pathNARR = extract_NARR_timeseries(cwd, path, basinName)
    outputNARR = NARR_format_timeseries_basin(output, pathNARR, startYear, endYear)
    dataGEO = extract_GIS_info(startYear, endYear, coords, cwd)
    tele = extract_tele_timeseries_basin(coords, cwd, startYear, endYear)
    dataALLKPN_seas, dataStats = all_data_format_condense(outputNARR, dataGEO, tele, cwd)
    stackPI, binaryPI = simulate_Kpn(dataALLKPN_seas, dataStats, cwd)
    write_Kpn_coords(stackPI, path, basinName)
    print("isoP has been completed!")

# The following two functions are for reading and writing the user profile. This is by far the least optimal way to do this
# but It works so I'm not going to change it. 
def readUser_Profile():
    # If the user has a profile file then read it in
    userProfile = {}
    with open("userProfile.txt", 'r') as f:
        for line in f:
            line = line.split(" = ")
            userProfile[line[0]] = line[1].strip()
    return userProfile

def writeUser_Profile(userInfo):
    # Save their profile then write it to a file
    if userInfo == None:
        print("""
            So the program detected that there was no userProfile.txt file and so it tried to make one. However, it did not save the information.
            I do not know why, and god help you.""")
    else:
        userInfo["userName"] = input("\nPlease enter your name to save the profile: ")
        with open("userProfile.txt", 'w') as f:
            f.write(f"userName = " + userInfo["userName"] + "\n")
            f.write(f"basinPath = " + userInfo["basinPath"] + "\n")
            f.write(f"basinName = " + userInfo["basinName"] + "\n")
            f.write(f"startYear = " + str(userInfo["startYear"]) + "\n")
            f.write(f"endYear = " + str(userInfo["endYear"]) + "\n")
            f.write(f"createdOn = " + datetime.now().strftime("%Y/%m/%d") + "\n")
        print("User profile has been saved! Now you will not have to enter the information again.")

def CLI_menu(cwd):
    inMenu = True
    if os.path.exists("userProfile.txt"):
        userProfile = readUser_Profile()
    else:
        userProfile = None

    while inMenu:
        print("\nPlease select an option from the following list:")
        print("------------------------------------------------")
        print("1. Instructions on how to run isoP")
        print("2. Download NARR data - Must be done before running isoP")
        print("3. Run isoP with WATFLOOD")
        print("4. Run isoP off of coordinates file only - No WATFLOOD output")
        print("5. Exit")
        userChoice = input("Enter the number of the option you would like to select: ")
    
        if userChoice == '1':
            print("Instructions on how to run isoP!")
            print("You must have a directory containing the following folders:")
            print("1. A basin folder that has the same name as the basin you are running isoP for (Doesn't matter what the name is as long as you are consistent with the basin name).")
            print("2. Within folder 1 an empty folder called 'isoP'.")
            print("*    This is only relevant for the non-WATFLOOD option.")
            print("*    For the non-WATFLOOD option you must also have a csv file (no header) with the coordinates you'd like to run isoP for.")
            print("*    The file must be named 'basinName_coords.csv' and be in the isoP folder.")
            print("3. After this you must run the program and select option 2 to download the NARR data if you haven't already.")
            print("*    The program will not run without the NARR data. They must also be updated periodically.")
            print("*    The teleconnection indices at this point need to be updated manually in the program folder")
            print("4. Then choose which ever option you would like to run isoP.")
            print("*    It will automatically write the output to the basin folder mentioned in step 1.")
            print("If you rerun the program IT WILL OVERWRITE the previous output files.\n")
        
        elif userChoice == '2':
            print("Downloading NARR data, this may take a while.")
            download_NARR_data()
            print("NARR data has been downloaded!")
        
        elif userChoice == '3':
            isoP_WATFLOOD(cwd, userProfile)
        
        elif userChoice == '4':
            isoP_coords(cwd, userProfile)
        
        elif userChoice == '5':
            print("Exiting the program.")
            inMenu = False
        
        else:
            print("Invalid input please try again.")
    

def main():
    # Currently just running the isoP function but will open the CLI menu when ready
    print("""
        ██ ███████  ██████  ██████  
        ██ ██      ██    ██ ██   ██ 
        ██ ███████ ██    ██ ██████  
        ██      ██ ██    ██ ██      
        ██ ███████  ██████  ██      
                       """)
    print("Written by C. Delavau December 2014. (c)")
    print("Updated by T. Holmes June 2017 for running isoP without WATFLOOD.")
    print("Converted to Python by J. Gray August 2023.")

    print("""WARNING: The python version is still in development. Currently the program cannot not perform the statistics, 
          so when asked if you would like to run the PI please select no.\n""")
    cwd = os.getcwd() # For openning files within the isoP folder and thus removing complications with file paths
    CLI_menu(cwd)
    
    
main()
