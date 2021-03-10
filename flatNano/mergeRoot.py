import os
import subprocess
import pandas as pd
from personal_settings import *

path_or = personal_tier_path+'dataframes_2021Mar08/'

dataset = 'BcToJPsiMuMu/'

#sistema per dati, non funziona piu mi sa !
if not ("BcToJPsiMuMu") in dataset:
    flag_names = ['ptmax']

else:
    flag_names = ['is_jpsi_tau','is_jpsi_mu','is_jpsi_pi','is_psi2s_mu','is_chic0_mu','is_chic1_mu','is_chic2_mu','is_hc_mu','is_psi2s_tau','is_jpsi_3pi','is_jpsi_hc']

for flag in flag_names:
    path = path_or
    
    #    if("BcToJPsiMuMu") in dataset:
    if ("BcToJPsiMuMu") in dataset:
        lsOut = subprocess.getoutput('ls ' + path + dataset + flag)
    else:
        lsOut = subprocess.getoutput('ls ' + path + dataset )
    print(lsOut)
    files = lsOut.split('\n')
    
    print("%s files are going to me merged" %(len(files)))
    
    hadd_opts = ''

    if(len(files) <=1000):
        for fil in files:
            #print(fil)
            if ("BcToJPsiMuMu") in dataset:
                hadd_opts = hadd_opts + path + dataset + flag + '/' +fil+ ' '
            else:
                hadd_opts = hadd_opts + path + dataset +  '/' +fil+ ' '

        os.system('hadd '+ path_or + dataset.strip('/') + '_'+flag + '_merged.root '+hadd_opts)        

    else:
        x = 1000
        print(len(files),int(len(files)/1000) +1)
        for y in range(int(len(files)/1000) + 1):
            print(y,x)
            if(x>len(files)):
                x = len(files)
            print(y*1000,x)
            hadd_opts = ''
            for fil in files[y*1000:x]:
                #print(fil)
                if ("BcToJPsiMuMu") in dataset:
                    hadd_opts = hadd_opts + path + dataset + flag + '/' +fil+ ' '
                else:
                    hadd_opts = hadd_opts + path + dataset +  '/' +fil+ ' '

            if(y == 0):
                os.system('hadd '+ path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y)+'.root '+hadd_opts)
            else:
                if(x!=len(files)):
                    os.system('hadd '+ path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y)+'.root '+hadd_opts+' '+path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y-1)+'.root')
                    os.system('rm -f '+path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y-1)+'.root')
                else:
                    os.system('hadd '+ path_or + dataset.strip('/') + '_'+flag + '_merged.root '+hadd_opts+' '+path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y-1)+'.root')
                    os.system('rm -f '+path_or + dataset.strip('/') + '_'+flag + '_merged_'+str(y-1)+'.root')
            
            x+=1000
    #    print(path_or + dataset.strip('/') + '_'+flag + '_merged.root Saved')
    
    
