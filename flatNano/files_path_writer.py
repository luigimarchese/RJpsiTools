################################################################################
## With this script it is possible to save in a .txt file the paths of nanoAODs produced with CRAB jobs
## To personalize it change the variables:
## personal_rjpsi_path -> path to your nanoaod producer
## username -> your username
## tier_access_path -> string to access the tier in which you saved the nanoaod files
################################################################################

################################################################################
## To run the script:
## cmsenv; activate crab; 'python files_path_write.py <dataset_name1>,<dataset_name2>
################################################################################


import subprocess
import os
import sys

nanotools = True

tier_access_path = 'srm://storage01.lcg.cscs.ch:8443/srm/managerv2?SFN=/pnfs/lcg.cscs.ch/cms/trivcat/store/user/friti' #CSCS
#tier_access_path = 'root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/friti' #PSI

personal_rjpsi_path = '/work/friti/rjpsi/CMSSW_10_6_14/src/PhysicsTools/RJPsiNano/'
username = 'friti'
jobs_date = '2021Mar16'

crab_path = personal_rjpsi_path + 'production/RJPsiNANO_'+jobs_date

dataset_dict = {
    'BcToXToJpsi':['BcToXToJpsi1','BcToXToJpsi2'],
    'data':['data_Run2018A_UL','data_Run2018B_UL','data_Run2018C_UL','data_Run2018D_UL'],
    'BcToJpsiMuNu':['BcToJpsiMuNu'],
    'BcToJpsiTauNu':['BcToJpsiTauNu'],
    'OniaX':['OniaX'],
    'BToJpsi_ToMuMu':['BToJpsi_ToMuMu'],
    'BcToJPsiMuMu':['BcToJPsiMuMu'],
    'HbToJPsiMuMu':['HbToJPsiMuMu'],
    'HbToJPsiMuMu_3MuFilter':['HbToJPsiMuMu_3MuFilter']
                }
#in input the dataset names separeted by a comma, without any blank space
datasets = map(str,sys.argv[1].split(','))
 
for dataset in datasets:
    print(" ")
    print("Dataset: %s"%dataset)
    if not os.path.exists("txt_files/"):
        os.makedirs("txt_files/")
    f = open("txt_files/"+dataset+"_files_path.txt", "w")
    folders = dataset_dict[dataset]
    for folder in folders:
        url = os.popen('crab getoutput --xrootd --jobids=1 -d ' + crab_path + '/crab_' + folder + '/').readlines()[0]
        print(crab_path + '/crab_' + folder)
        print(url)
        s1 = url.split(username)
        s2 = s1[1].split('0000')
        newurl = tier_access_path + s2[0]

        i = 0
        print('Checking files in the folder '+newurl.strip('\n'))
        while True:
            files_name = os.popen('eval `scram unsetenv -sh`; gfal-ls '+ newurl.strip('\n')+'000'+str(i)).readlines()
            path_name = newurl.strip('\n')+'000'+str(i)
            if(len(files_name)==0):
                print("The folder does not exist: "+ str(path_name))
                break
            print('subfolder: '+'000'+str(i))
            for file in range(len(files_name)):
                if(files_name[file].strip('\n') == 'log'):
                        continue
                #                print('root://cms-xrd-global.cern.ch//store/user/'+username+path_name.split(tier_access_path)[1]+'/'+files_name[file])
                f.write('root://cms-xrd-global.cern.ch//store/user/'+username+path_name.split(tier_access_path)[1]+'/'+files_name[file]) 
            i+=1

    f.close()
    print("The file txt_files/"+dataset+"_files_path.txt has been created.")
    
print('\nGoodbye\n')
