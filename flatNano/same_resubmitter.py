## this resubmitted resubmit the failed jobs as they were (without any more splitting)
import os
from personal_settings import *

#data, HbToJPsiMuMu, BcToJPsiMuMu
#dataset = 'BcToJPsiMuMu'
dataset = 'HbToJPsiMuMu'
dateFolder = '2021Mar08'
#already resubmitted once

#res = False
res = True
out_dir = "dataframes_"+ dateFolder+ "/"+dataset

print("The dataset is ",dataset)

if(res == False):
    fin = open("dataframes_"+dateFolder+"/"+dataset+"/"+dataset+"_files_check.txt","r")
else:
    fin = open("dataframes_"+dateFolder+"/"+dataset+"/"+dataset+"_files_check_resubmitted.txt","r")

list_of_expected_files = [s.split(' \n')[0] for s in fin.readlines()]

if (not 'BcToJPsiMuMu' in dataset):
    list_of_created_files = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset).readlines()
    list_of_created_files = [s.split('\n')[0] for s in list_of_created_files]
    for fl in list_of_expected_files:
        if fl not in list_of_created_files:
            i = fl.split('_')[2]
            if(res == True):
                j = fl.split('_')[3]
                command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%s_%s.log -e %s/errs/chunk%s_%s.err --job-name=%s_res  %s/submitter_chunk%s_%s.sh' %(out_dir, i,j, out_dir, i,j, dataset, out_dir, i,j)
            else:
                command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%s.log -e %s/errs/chunk%s.err --job-name=%s_res  %s/submitter_chunk%s.sh' %(out_dir, i, out_dir, i, dataset, out_dir, i)

            os.system(command_sh_batch)
            

if('BcToJPsiMuMu' in dataset):
    list_of_created_jpsi_tau = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_tau").readlines()
    list_of_created_jpsi_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_mu").readlines()
    list_of_created_jpsi_pi = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_pi").readlines()
    list_of_created_jpsi_hc = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_hc").readlines()
    list_of_created_jpsi_3pi = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_jpsi_3pi").readlines()
    list_of_created_chic0_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic0_mu").readlines()
    list_of_created_chic1_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic1_mu").readlines()
    list_of_created_chic2_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_chic2_mu").readlines()
    list_of_created_hc_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_hc_mu").readlines()
    list_of_created_psi2s_mu = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_psi2s_mu").readlines()
    list_of_created_psi2s_tau = os.popen('ls '+personal_tier_path + 'dataframes_'+ dateFolder + '/'+ dataset+"/is_psi2s_tau").readlines()

    list_of_created_jpsi_tau = [s.split('\n')[0] for s in list_of_created_jpsi_tau]
    list_of_created_jpsi_mu = [s.split('\n')[0] for s in list_of_created_jpsi_mu]
    list_of_created_jpsi_pi = [s.split('\n')[0] for s in list_of_created_jpsi_pi]
    list_of_created_jpsi_hc = [s.split('\n')[0] for s in list_of_created_jpsi_hc]
    list_of_created_jpsi_3pi = [s.split('\n')[0] for s in list_of_created_jpsi_3pi]
    list_of_created_chic0_mu = [s.split('\n')[0] for s in list_of_created_chic0_mu]
    list_of_created_chic1_mu = [s.split('\n')[0] for s in list_of_created_chic1_mu]
    list_of_created_chic2_mu = [s.split('\n')[0] for s in list_of_created_chic2_mu]
    list_of_created_hc_mu = [s.split('\n')[0] for s in list_of_created_hc_mu]
    list_of_created_psi2s_mu = [s.split('\n')[0] for s in list_of_created_psi2s_mu]
    list_of_created_psi2s_tau = [s.split('\n')[0] for s in list_of_created_psi2s_tau]
    
    resub = []
    for fl in list_of_expected_files:
        flag = 0
        if ('is_jpsi_tau' in fl ) and (fl not in list_of_created_jpsi_tau):
            flag = 1
        if ('is_jpsi_mu' in fl ) and (fl not in list_of_created_jpsi_mu):
            flag = 1
        if ('is_jpsi_pi' in fl ) and (fl not in list_of_created_jpsi_pi):
            flag = 1
        if ('is_jpsi_hc' in fl ) and (fl not in list_of_created_jpsi_hc):
            flag = 1
        if ('is_jpsi_3pi' in fl ) and (fl not in list_of_created_jpsi_3pi):
            flag = 1
        if ('is_chic0_mu' in fl ) and (fl not in list_of_created_chic0_mu):
            flag = 1
        if ('is_chic1_mu' in fl ) and (fl not in list_of_created_chic1_mu):
            flag = 1
        if ('is_chic2_mu' in fl ) and (fl not in list_of_created_chic2_mu):
            flag = 1
        if ('is_hc_mu' in fl ) and (fl not in list_of_created_hc_mu):
            flag = 1
        if ('is_psi2s_mu' in fl ) and (fl not in list_of_created_psi2s_mu):
            flag = 1
        if ('is_psi2s_tau' in fl ) and (fl not in list_of_created_psi2s_tau):
            flag = 1
        
        i = fl.split('_')[2]
        if flag == 1:
            if len(resub) ==0:
                resub.append(i)
            else:
                if(resub[-1]!=i):
                    resub.append(i)

    for i in resub:
        print("======> PROCESSING file n ",i)
        command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%s.log -e %s/errs/chunk%s.err --job-name=%s_res --mem=4GB  %s/submitter_chunk%s.sh' %(out_dir, i, out_dir, i, dataset, out_dir, i)
            
        os.system(command_sh_batch)
