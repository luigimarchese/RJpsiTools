'''
Submitter to the psi tier3 batch https://wiki.chipp.ch/twiki/bin/view/CmsTier3/SlurmUsage
for processing nanoaod files

Difference with v3:
- fixed for other samples other than Bc

Difference from v2:
- submits splitted channels for Bc
'''
import os
import datetime
import sys
from personal_settings import *

# nanoaod datasets names and the corresponding files
dataset_dict = {
    'data':                  ['--data '+personal_tools_path+'/txt_files/data_files_path_2021Sep30.txt'],
    'BcToJPsiMuMu':          ['--mc_bc '+personal_tools_path+'/txt_files/BcToJPsiMuMu_files_path_2021Sep21.txt'],
    'BcToJpsiTauNu':         ['--mc_tau '+personal_tools_path+'/txt_files/BcToJpsiTauNu_files_path_2021Sep21.txt'],
    'HbToJPsiMuMu':          ['--mc_hb '+personal_tools_path+'/txt_files/HbToJPsiMuMu_files_path_2021Sep21.txt'],
    'HbToJPsiMuMu_3MuFilter':['--mc_hb '+personal_tools_path+'/txt_files/HbToJPsiMuMu_3MuFilter_files_path_2021Sep21.txt'],
    'BuToJpsiK':             ['--mc_hb '+personal_tools_path+'/txt_files/BuToJpsiK_files_path_2021Dec08.txt'],
}

dataset = 'BuToJpsiK'
dataset_opt = dataset_dict[dataset][0]

file_name = dataset_opt.split(' ')[1]
count_files = len(open(file_name).readlines(  ))

files_per_job = 25
njobs = count_files//files_per_job + 1  

print("Submitting %s jobs" %(njobs))
production_tag = datetime.date.today().strftime('%Y%b%d')

date_dir = 'dataframes_'+ production_tag
out_dir = date_dir + '/%s' %(dataset)

bc_samples = [
    'is_chic0_mu',
    'is_chic1_mu',
    'is_chic2_mu',
    'is_hc_mu',
    'is_jpsi_3pi',
    'is_jpsi_hc',
    'is_jpsi_mu',
    'is_jpsi_tau',
    'is_jpsi_pi',
    'is_psi2s_mu',
    'is_psi2s_tau'
]

####################################################
####### Make the directories #######################
####################################################

# make output dir
if not os.path.exists(date_dir):
# if there si already one it does not delete it
    os.makedirs(date_dir)

#do not write in the same folder if I forget to change the name of the folder!
if os.path.exists(out_dir):
    sys.exit("WARNING: the folder "+ out_dir + " already exists!")
    
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')
os.system('cp nanoframe.py '+ out_dir+ '/.')
os.system('cp mybatch.py '+ out_dir+ '/.')
os.system('cp bgl_variations.py '+ out_dir+ '/.')
os.system('cp decay_weight.root '+ out_dir+ '/.')

fcheck = open(out_dir+"/"+dataset+"_files_check.txt","w+")

if not os.path.exists(personal_tier_path + date_dir):
    os.makedirs(personal_tier_path + date_dir)

if not os.path.exists(personal_tier_path +out_dir):
    os.makedirs(personal_tier_path +out_dir)
else:
    sys.exit("WARNING: the folder "+ out_dir + " already exists in the SE!")

# for Bc samples we need many folder as many dataset are contained in the sample
if ((("BcToXToJpsi") in dataset) or (("BcToJPsiMuMu") in dataset) ):
    print("Making Bc directories...")
    for bc_name in bc_samples:
        if not os.path.exists(personal_tier_path +out_dir+ "/"+bc_name):
            os.makedirs(personal_tier_path +out_dir + "/"+bc_name)

###################################################
#### Preparing for the submission #################
###################################################
for ijob in range(njobs):
    
    if ((not ("BcToXToJpsi") in dataset) and (not ("BcToJPsiMuMu") in dataset) ):
        channels = [['BTo3Mu','BTo2MuP','BTo2MuK','BTo2Mu3P']]
        name_add = ['']
        samples = ['ptmax']

    else:
        # in case of Bc sample, for unknown reasons, we need to split the channels because otherwise the jobs fail
        channels = [['BTo3Mu'],['BTo2MuP','BTo2MuK','BTo2Mu3P']]
        name_add = ['_3m','_others']
        samples = bc_samples
        
    for add, channel in zip(name_add,channels):
        #input file
        fin = open("Resonant_Rjpsi_v5_crab_splitted_channels.py", "rt")
        #output file to write the result to (name of the jobs+ subjob)
        fout = open("%s/Resonant_Rjpsi_chunk%d%s.py" %(out_dir, ijob, add), "wt")
        #for each line in the input file
        for line in fin:
            #read replace the string and write to output file
            if   'REPLACE_MAX_FILES' in line: fout.write(line.replace('REPLACE_MAX_FILES' , '%s' %files_per_job))
            elif 'REPLACE_CHANNELS'   in line: fout.write(line.replace('REPLACE_CHANNELS'   , '%s' %channel))
            elif 'REPLACE_FILE_OUT'   in line: fout.write(line.replace('REPLACE_FILE_OUT'   , '/scratch/friti/%s/%s_UL_%d%s' %(dataset, dataset,ijob,add)))
            elif 'REPLACE_SKIP_FILES'in line: fout.write(line.replace('REPLACE_SKIP_FILES', '%d' %(files_per_job*ijob)))
            else: fout.write(line)
        #close input and output files
        fout.close()
        fin.close()

        # each job needs a submitter script
        flauncher = open("%s/submitter_chunk%d%s.sh" %(out_dir, ijob,add), "wt")
        
        write_string = ''
        bash_check = ''
        for sample in samples:
            fcheck.write(dataset+"_UL_"+str(ijob)+add+"_"+sample+".root \n")

            if ((("BcToXToJpsi") in dataset) or (("BcToJPsiMuMu") in dataset) ):
                write_string += 'xrdcp /scratch/%s/%s/%s_UL_%s%s_%s.root root://t3dcachedb.psi.ch:1094///%s%s/%s/. \n'%(username, dataset, dataset, ijob, add, sample, personal_tier_path, out_dir, sample)
            else:
                write_string += 'xrdcp /scratch/%s/%s/%s_UL_%s%s_%s.root root://t3dcachedb.psi.ch:1094///%s%s/. \n'%(username, dataset, dataset, ijob, add, sample, personal_tier_path, out_dir)


            write_string += 'rm /scratch/%s/%s/%s_UL_%s%s_%s.root \n'%(username, dataset, dataset, ijob, add, sample)


            #https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php
            '''if add == '_3mu':
                bash_check += 'test -e /pnfs/psi.ch/cms/trivcat/%s%s/%s/%s_UL_%s_others_%s.root \n'%(personal_tier_path,out_dir, sample,file_name, dataset, ijob,  sample)
            else::
                bash_check += 'test -e /pnfs/psi.ch/cms/trivcat/%s%s/%s/%s_UL_%s_3mu_%s.root \n'%(personal_tier_path,out_dir, sample,file_name, dataset, ijob,  sample)
                
            bash_check += 'if [$? = 0]\n'
            bash_check += 'then \n'
            bash_check += 'hadd  /pnfs/psi.ch/cms/trivcat/%s%s/%s/%s_UL_%s_%s.root /pnfs/psi.ch/cms/trivcat/%s%s/%s/%s_UL_%s_3mu_%s.root /pnfs/psi.ch/cms/trivcat/%s%s/%s/%s_UL_%s_others_%s.root\n'%(personal_tier_path,out_dir, sample,file_name, dataset, ijob, sample, username, dataset, dataset, ijob,add, sample, personal_tier_path, out_dir, sample, personal_tier_path,out_dir, sample,file_name, dataset, ijob,  sample)
            '''
        flauncher.write(
            '''#!/bin/bash
            cd {dir}
            #scramv1 runtime -sh
            mkdir -p /scratch/{username}/{scratch_dir}
            ls /scratch/{username}/
            python {cfg} {option}
            ls /scratch/{username}/{scratch_dir}
            {string}'''.format(dir='/'.join([os.getcwd(), out_dir]), username= username,tier3_path=personal_tier_path,scratch_dir= dataset, cfg='Resonant_Rjpsi_chunk%d%s.py' %(ijob, add), option= dataset_opt, dat = dataset,ijob=ijob, se_dir=out_dir, string = write_string))
            
            

        flauncher.close()
        #command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%d.log -e %s/errs/chunk%d.err --job-name=%s --time=60 --mem=6GB %s/submitter_chunk%d.sh' %(out_dir, ijob, out_dir, ijob, out_dir, out_dir, ijob)
        command_sh_batch = 'sbatch -p long --account=t3 --mem=5G -o %s/logs/chunk%d%s.log -e %s/errs/chunk%d%s.err --job-name=%s   %s/submitter_chunk%d%s.sh ' %(out_dir, ijob, add, out_dir, ijob, add, dataset, out_dir, ijob, add)
        #--mem=6GB
        #print(command_sh_batch)
            
        os.system(command_sh_batch)

fcheck.close()
    
