'''
Script to submit the inspector mu on the batch
'''
import os
from glob import glob

decay = 'mu'
input_directory = '/RJPsi_Bc_GEN_23May22_v7/'

out_dir = 'Rjpsi_inspector_bc_'+decay+'_23May22_v7'

template_inspector = "inspector_"+decay+"_TEMPLATE.py"
template_fileout = "RJpsi_inspector_bc_"+decay+"_TEMPLATE.root"

##########################################################################################
##########################################################################################

# make output dir
if not os.path.exists(out_dir):
    try:
        os.makedirs('/pnfs/psi.ch/cms/trivcat/store/user/friti/'+out_dir)
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')
#os.system('cp files_HbToJPsiMuMu_3MuFilter_old.py ' + out_dir + '/.')
#os.system('cp -r GeneratorInterface ' + out_dir + '/.')
events_per_job = -1

files_name = os.popen('ls /pnfs/psi.ch/cms/trivcat/store/user/friti/RJPsi_Bc_GEN_23May22_v7/')
jobs = []
for file in files_name:
    ijob=file.split("_")[1]
    if ijob not in jobs:
        jobs.append(ijob)

for ijob in jobs:
    files = glob('/pnfs/psi.ch/cms/trivcat/store/user/friti/RJPsi_Bc_GEN_23May22_v7/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_%s_*.root'%ijob)
    #print(str(files))
    tmp_inspector = template_inspector.replace('TEMPLATE', 'chunk%s' %ijob)
    tmp_fileout = template_fileout.replace('TEMPLATE', '%s'%ijob)
    
    #input file
    fin = open(template_inspector, "rt")
    #output file to write the result to
    fout = open("%s/%s" %(out_dir, tmp_inspector), "wt")
    #for each line in the input file
    for line in fin:
        #read replace the string and write to output file
        if   'HOOK_FILE_IN'    in line: fout.write(line.replace('HOOK_FILE_IN'   , str(files)))
        elif   'HOOK_INPUT'    in line: fout.write(line.replace('HOOK_INPUT'   , input_directory))
        elif 'HOOK_MAX_EVENTS' in line: fout.write(line.replace('HOOK_MAX_EVENTS', '%d' %events_per_job))
        elif 'HOOK_SKIP_EVENTS' in line: fout.write(line.replace('HOOK_SKIP_EVENTS', '0' ))

        elif 'HOOK_FILE_OUT'   in line: fout.write(line.replace('HOOK_FILE_OUT'  , '/scratch/friti/%s/%s' %(out_dir, tmp_fileout)))
        else: fout.write(line)
    #close input and output files
    fout.close()
    fin.close()

    to_write = '\n'.join([
        '#!/bin/bash',
        'cd {dir}',
        'scramv1 runtime -sh',
        'mkdir -p /scratch/friti/{scratch_dir}',
        'ls /scratch/friti/',
        #'python {insp} --verbose 0 --files_per_job {files_per_job} --jobid {jobid}',
        'python {insp} --verbose 0 ',
        'xrdcp /scratch/friti/{scratch_dir}/{fout} root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/friti/{se_dir}/{fout}',
        'rm /scratch/friti/{scratch_dir}/{fout}',
        'echo {fout} Saved!',
        '',
    ]).format(
        dir           = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir   = out_dir, 
        insp           = tmp_inspector, 
        #files_per_job = files_per_job,
        se_dir        = out_dir,
        #jobid         = ijob,
        fout          = tmp_fileout
        )

    with open("%s/submitter_chunk%s.sh" %(out_dir, ijob), "wt") as flauncher: 
        flauncher.write(to_write)
    
    command_sh_batch = ' '.join([
        'sbatch', 
        #'-p testnew', 
        '--account=t3', 
        '-o %s/logs/chunk%s.log' %(out_dir, ijob),
        '-e %s/errs/chunk%s.err' %(out_dir, ijob), 
        '--job-name=%s_insp' %str(ijob), 
        #'--time=60', 
        '%s/submitter_chunk%s.sh' %(out_dir, ijob), 
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)

