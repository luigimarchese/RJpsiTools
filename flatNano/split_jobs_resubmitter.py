## This resubmitter can split the jobs that failed into smaller jobs
## It doesn't work with BcToJpsiMuMu
import os
import datetime
import sys
from personal_settings import *

dataset_dict = {'data':['--data '+personal_tools_path+'/txt_files/data_files_path.txt'],
                #                'BcToJPsiMuMu':['--mc_bc '+personal_tools_path+'/txt_files/BcToJPsiMuMu_files_path.txt'],
                'HbToJPsiMuMu':['--mc_hb '+personal_tools_path+'/txt_files/HbToJPsiMuMu_files_path.txt']
}

dataset = 'HbToJPsiMuMu'
dateFolder = '2021Mar08'
new_files_per_job = 1
dataset_opt = dataset_dict[dataset][0]

out_dir = "dataframes_"+ dateFolder+ "/"+dataset

#total files supposed to be produced with the first submission
# number taken from .txt file produced during the first submission
f_n_files = open(personal_tools_path +"dataframes_"+ dateFolder+ "/"+dataset + "/"+dataset+"_files_check.txt","r")
total_files = len(f_n_files.readlines())
f_n_files.close()
print("The total expected files were ",total_files)

files_ok = os.popen('ls ' +personal_tier_path+ 'dataframes_' + dateFolder + '/'+ dataset).readlines()

numbers_ok = []
for fil in files_ok:
    #    print(fil)
    numbers_ok.append(int(fil.split("_")[2]))
#print(numbers_ok)
numbers_ok.sort()
print("Files already processed",numbers_ok)

#file to write in all the files I should obtain in the end in my folder, to check
fcheck = open(out_dir+"/"+dataset+"_files_check_resubmitted.txt","w+")

for i in range(total_files):
#for i in range(1):
    if(i in numbers_ok):
        if((not ("BcToXToJpsi") in dataset) and (not ("BcToJPsiMuMu") in dataset)):
            fcheck.write(dataset+"_UL_"+str(i)+"_ptmax.root \n")
    if(i not in numbers_ok):
        print("======> PROCESSING file n ",i)
        fin = open(personal_tools_path + "dataframes_"+ dateFolder+ "/"+dataset+"/Resonant_Rjpsi_chunk"+str(i)+".py", "rt")
        lines = fin.readlines()
        files_per_job = 0
        skip_files = 0
        for line in lines[29:32]:
            if 'nMaxFiles' in line: files_per_job = int(line.split(" ")[2])
            if 'skipFiles' in line: skip_files = int(line.split(" ")[2])
        #input file
        fin.close()

        #we have to send new jobs!
        fin2 = open("Resonant_Rjpsi_v3_crab.py", "rt")
        lines = fin2.readlines()

        #output file to write the result in
        print("Old files per job",files_per_job,"New files per job",new_files_per_job)
        if(new_files_per_job <= files_per_job and files_per_job%new_files_per_job==0 ):
            nNewJobs= (files_per_job//new_files_per_job) 
        for j in range(nNewJobs):
            fout = open(personal_tools_path + "dataframes_"+ dateFolder+ "/"+dataset+"/Resonant_Rjpsi_chunk%d_%d.py" %(i, j), "wt")

            #for each line in the input file
            #            print("Creating "+personal_tools_path +"dataframes_"+ dateFolder+ "/"+dataset+"/Resonant_Rjpsi_chunk%d_%d.py" %(i, j))
            for line in lines:
                #read replace the string and write to output file
                if 'REPLACE_MAX_FILES' in line: fout.write(line.replace('REPLACE_MAX_FILES' , '%s' %(new_files_per_job)))
                elif 'REPLACE_FILE_OUT'   in line: fout.write(line.replace('REPLACE_FILE_OUT'   , '/scratch/friti/%s/%s_UL_%d_%d' %(dataset, dataset,i,j)))
                elif 'REPLACE_SKIP_FILES'in line: fout.write(line.replace('REPLACE_SKIP_FILES', '%d' %(skip_files + new_files_per_job*j)))
                else: fout.write(line)
                #close input and  output files

            fout.close()
            fin2.close()
            
            flauncher = open(personal_tools_path +"dataframes_"+ dateFolder+ "/"+dataset+"/submitter_chunk%d_%d.sh" %(i,j), "wt")
            if ((not("BcToXToJpsi") in dataset) and (not ("BcToJPsiMuMu") in dataset)):
               
                flauncher.write(
                    '''#!/bin/bash
                    cd {dir}
                    #scramv1 runtime -sh
                    mkdir -p /scratch/{username}/{scratch_dir}
                    ls /scratch/{username}/
                    python {cfg} {option}
                    ls /scratch/{username}/{scratch_dir}
                    xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{i}_{j}_ptmax.root root://t3dcachedb.psi.ch:1094///{tier3_path}{se_dir}/.
                    rm /scratch/{username}/{scratch_dir}/{dat}_UL_{i}_{j}_ptmax.root'''.format(dir = out_dir, username= username,tier3_path=personal_tier_path,scratch_dir= dataset, cfg='Resonant_Rjpsi_chunk%d_%d.py' %(i,j), option= dataset_opt, dat = dataset,i=i,j=j, se_dir=out_dir)
                )
                fcheck.write(dataset+"_UL_"+str(i)+"_"+str(j)+"_ptmax.root \n")

                
            flauncher.close()

            command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%d_%d.log -e %s/errs/chunk%d_%d.err --job-name=%s_res --mem=6G  %s/submitter_chunk%d_%d.sh' %(out_dir, i,j, out_dir, i,j, dataset, out_dir, i,j)
            #            print(command_sh_batch)
            os.system(command_sh_batch)
fcheck.close()
            
