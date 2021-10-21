'''
Script that reads filter efficiencies and the time per event from log files of MC production, to compute the average
'''
import numpy as np
import os

# Which mc productions
folders = [
    'RJpsi-HbToJpsiMuMu-3MuFilter_GEN_11Oct_S1_v1', # ScaleToFilter 1
    'RJpsi-HbToJpsiMuMu-3MuFilter_GEN_27Sep21_v7',  # ScaleToFilter 3
    'RJpsi-HbToJpsiMuMu-3MuFilter_GEN_27Sep21_v5',  # ScaleToFilter 5
]


#initialize array for filter eff and for time
filter_efficiency = []
filter_efficiency_unc = []


time_per_event = []

#files = files[:2]
# Open one file per time to read the two interesting informations
# It needs some string manipulation

for SF,folder in zip([1,3,5],folders):
    # path -> the info is in the err log
    path = '/work/friti/genMC/CMSSW_10_6_20/src/' + folder + '/errs/'

    #read the name of all the log files in the folder
    files_string = os.popen('ls ' + path).readlines()
    files = [file.strip('\n') for file in files_string]

    for file in files:
        f = open(path + '/' + file, 'r')
        lines = f.readlines()
        for line in lines:
            if 'Filter efficiency (event-level)=' in line:
                saved_line = line.replace("Filter efficiency (event-level)=","")
                saved_line = saved_line.replace("    [TO BE USED IN MCM]","")
                saved_line = saved_line.split("=")
                saved_line = saved_line[1].split("+-")
                filter_efficiency.append(float(saved_line[0].replace(" ","")))
                filter_efficiency_unc.append(float(saved_line[1].replace(" ","")))
            if '- Avg event:' in line:
                saved_line = line.replace('- Avg event:','')
                saved_line = saved_line.strip(' ')
                time_per_event.append(float(saved_line))
    
    #Compute the weighted average for the filter efficiency
    # media pesata? O media?
    filter_eff_weighted = [(eff*unc) for eff,unc in zip(filter_efficiency,filter_efficiency_unc)] 
    average = np.sqrt(sum(filter_eff_weighted)/len(filter_eff_weighted))
    print("Filter efficiency for ScaleToFilter " + str(SF) + " is: " + str(sum(filter_efficiency)/len(filter_efficiency)))
    print("Average time per event for ScaleToFilter " + str(SF) + " is: " + str(sum(time_per_event)/len(time_per_event)))
    #print(average)


