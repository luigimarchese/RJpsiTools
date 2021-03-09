import os
import datetime
import sys
from personal_settings import *

dataset_dict = {'data':['--data '+personal_tools_path+'/txt_files/data_files_path.txt'],
                'BcToJPsiMuMu':['--mc_bc '+personal_tools_path+'/txt_files/BcToJPsiMuMu_files_path.txt'],
                'HbToJPsiMuMu':['--mc_hb '+personal_tools_path+'/txt_files/HbToJPsiMuMu_files_path.txt']
}

dataset = 'BcToJPsiMuMu'
dataset_opt = dataset_dict[dataset][0]

#splitting of files
file_name = dataset_opt.split(' ')[1]
count_files = len(open(file_name).readlines(  ))

files_per_job = 1
njobs = count_files//files_per_job + 1  

print("Submitting %s jobs" %(njobs))
#print("Submitting in quick.")
production_tag = datetime.date.today().strftime('%Y%b%d')

date_dir = 'dataframes_'+ production_tag
out_dir = date_dir + '/%s' %(dataset)


##########################################################################################
##########################################################################################

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

fcheck = open(out_dir+"/"+dataset+"_files_check.txt","w+")

if not os.path.exists(personal_tier_path + date_dir):
    os.makedirs(personal_tier_path + date_dir)

if not os.path.exists(personal_tier_path +out_dir):
    os.makedirs(personal_tier_path +out_dir)
else:
    sys.exit("WARNING: the folder "+ out_dir + " already exists in the SE!")

if ((("BcToXToJpsi") in dataset) or (("BcToJPsiMuMu") in dataset)):
    print("Making Bc directories...")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_hc_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_hc_mu")

    if not os.path.exists(personal_tier_path +out_dir+ "/is_chic0_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_chic0_mu")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_chic1_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_chic1_mu")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_chic2_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_chic2_mu")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_jpsi_3pi"):
        os.makedirs(personal_tier_path +out_dir + "/is_jpsi_3pi")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_jpsi_hc"):
        os.makedirs(personal_tier_path +out_dir + "/is_jpsi_hc")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_jpsi_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_jpsi_mu")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_jpsi_pi"):
        os.makedirs(personal_tier_path +out_dir + "/is_jpsi_pi")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_jpsi_tau"):
        os.makedirs(personal_tier_path +out_dir + "/is_jpsi_tau")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_psi2s_mu"):
        os.makedirs(personal_tier_path +out_dir + "/is_psi2s_mu")
    if not os.path.exists(personal_tier_path +out_dir+ "/is_psi2s_tau"):
        os.makedirs(personal_tier_path +out_dir + "/is_psi2s_tau")

for ijob in range(njobs):
    #input file
    fin = open("Resonant_Rjpsi_v3_crab.py", "rt")
    #output file to write the result to
    fout = open("%s/Resonant_Rjpsi_chunk%d.py" %(out_dir, ijob), "wt")
    #for each line in the input file
    for line in fin:
        #read replace the string and write to output file
        if   'REPLACE_MAX_FILES' in line: fout.write(line.replace('REPLACE_MAX_FILES' , '%s' %files_per_job))
        elif 'REPLACE_FILE_OUT'   in line: fout.write(line.replace('REPLACE_FILE_OUT'   , '/scratch/friti/%s/%s_UL_%d' %(dataset, dataset,ijob)))
        elif 'REPLACE_SKIP_FILES'in line: fout.write(line.replace('REPLACE_SKIP_FILES', '%d' %(files_per_job*ijob)))
        else: fout.write(line)
    #close input and output files
    fout.close()
    fin.close()

    flauncher = open("%s/submitter_chunk%d.sh" %(out_dir, ijob), "wt")
    if ((not ("BcToXToJpsi") in dataset) and (not ("BcToJPsiMuMu") in dataset)):
        fcheck.write(dataset+"_UL_"+str(ijob)+"_ptmax.root \n")
        flauncher.write(
            '''#!/bin/bash
            cd {dir}
            #scramv1 runtime -sh
            mkdir -p /scratch/{username}/{scratch_dir}
            ls /scratch/{username}/
            python {cfg} {option}
            ls /scratch/{username}/{scratch_dir}
            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_ptmax.root root://t3dcachedb.psi.ch:1094///{tier3_path}{se_dir}/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_ptmax.root'''.format(dir='/'.join([os.getcwd(), out_dir]), username= username,tier3_path=personal_tier_path,scratch_dir= dataset, cfg='Resonant_Rjpsi_chunk%d.py' %(ijob), option= dataset_opt, dat = dataset,ijob=ijob, se_dir=out_dir)
        )
    else:
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_chic0_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_chic1_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_chic2_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_hc_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_jpsi_3pi.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_jpsi_hc.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_jpsi_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_jpsi_tau.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_jpsi_pi.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_psi2s_mu.root \n")
        fcheck.write(dataset+"_UL_"+str(ijob)+"_is_psi2s_tau.root \n")


        flauncher.write(
            '''#!/bin/bash
            cd {dir}
            #scramv1 runtime -sh
            mkdir -p /scratch/{username}/{scratch_dir}
            ls /scratch/{username}/
            python {cfg} {option}
            ls /scratch/{username}/{scratch_dir}
            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic0_mu.root root://t3dcachedb.psi.ch:1094///{personal_tier_path}/{se_dir}/is_chic0_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic0_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic1_mu.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_chic1_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic1_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic2_mu.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_chic2_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_chic2_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_hc_mu.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_hc_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_hc_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_3pi.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_jpsi_3pi/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_3pi.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_hc.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_jpsi_hc/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_hc.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_mu.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_jpsi_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_pi.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_jpsi_pi/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_pi.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_tau.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_jpsi_tau/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_jpsi_tau.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_psi2s_mu.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_psi2s_mu/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_psi2s_mu.root

            xrdcp /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_is_psi2s_tau.root root://t3dcachedb.psi.ch:1094//{personal_tier_path}/{se_dir}/is_psi2s_tau/.
            rm /scratch/{username}/{scratch_dir}/{dat}_UL_{ijob}_psi2s_tau.root


            
            '''.format(dir='/'.join([os.getcwd(), out_dir]), scratch_dir= dataset, cfg='Resonant_Rjpsi_chunk%d.py' %(ijob), option= dataset_opt, username = username, personal_tier_path = personal_tier_path,dat = dataset,ijob=ijob, se_dir=out_dir)
        )
    
    flauncher.close()
    #command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%d.log -e %s/errs/chunk%d.err --job-name=%s --time=60 --mem=6GB %s/submitter_chunk%d.sh' %(out_dir, ijob, out_dir, ijob, out_dir, out_dir, ijob)
    command_sh_batch = 'sbatch -p wn --account=t3 -o %s/logs/chunk%d.log -e %s/errs/chunk%d.err --job-name=%s --mem=6GB   %s/submitter_chunk%d.sh' %(out_dir, ijob, out_dir, ijob, dataset, out_dir, ijob)
    #--mem=6GB
    #print(command_sh_batch)

    #os.system(command_sh_batch)

fcheck.close()
    
