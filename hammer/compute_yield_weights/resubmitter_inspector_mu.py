'''
Script to submit the inspector mu on the batch
'''
import os
from glob import glob
import ROOT

input_directory = '/RJPsi_Bc_GEN_23May22_v7/'

out_dir = 'Rjpsi_inspector_bc_mu_23May22_v7'

files_done = os.popen("ls /pnfs/psi.ch/cms/trivcat/store/user/friti/"+out_dir)
jobs_done=[]
for file in files_done:
    ijob=file.split("_")[4].split(".root")[0]
    if ijob not in jobs_done:
        jobs_done.append(ijob)

template_inspector = "inspector_mu_TEMPLATE.py"
template_fileout = "RJpsi_inspector_bc_mu_TEMPLATE.root"

##########################################################################################
##########################################################################################

#events_per_job = 

files_name = os.popen('ls /pnfs/psi.ch/cms/trivcat/store/user/friti/RJPsi_Bc_GEN_23May22_v7/')
jobs = []
for file in files_name:
    ijob=file.split("_")[1]
    if ijob not in jobs:
        jobs.append(ijob)

tot_jobs = 0
for ijob in jobs:
    if ijob in jobs_done:
        #print(ijob)
        continue
    files = glob('/pnfs/psi.ch/cms/trivcat/store/user/friti/RJPsi_Bc_GEN_23May22_v7/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_%s_*.root'%ijob)
    #print(str(files))
    #check nevents
    nevents = 0
    for file in files:
        gen_file = ROOT.TFile.Open(file,"r")
        nevents += gen_file.Get("Events").GetEntries()
        gen_file.Close()

    jjobs = 3
    events_per_job = nevents/jjobs

    for jjob in range(jjobs):
        
        tmp_inspector = template_inspector.replace('TEMPLATE', 'chunk%s_%d' %(ijob,jjob))
        tmp_fileout = template_fileout.replace('TEMPLATE', '%s_%d'%(ijob,jjob))
        #input file
        fin = open(template_inspector, "rt")
        #output file to write the result to
        fout = open("%s/%s" %(out_dir, tmp_inspector), "wt")
        #for each line in the input file
        skip_events = jjob*events_per_job
        for line in fin:
            #read replace the string and write to output file
            if   'HOOK_FILE_IN'    in line: fout.write(line.replace('HOOK_FILE_IN'   , str(files)))
            elif   'HOOK_INPUT'    in line: fout.write(line.replace('HOOK_INPUT'   , input_directory))
            elif 'HOOK_MAX_EVENTS' in line: fout.write(line.replace('HOOK_MAX_EVENTS', '%d' %events_per_job))
            elif 'HOOK_SKIP_EVENTS' in line: fout.write(line.replace('HOOK_SKIP_EVENTS', '%d' %skip_events))
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

        with open("%s/submitter_chunk%s_%d.sh" %(out_dir, ijob, jjob), "wt") as flauncher: 
            flauncher.write(to_write)
    
        command_sh_batch = ' '.join([
            'sbatch', 
            #'-p testnew', 
            '--account=t3', 
            '-o %s/logs/chunk%s_%d.log' %(out_dir, ijob, jjob),
            '-e %s/errs/chunk%s_%d.err' %(out_dir, ijob, jjob), 
            '--job-name=%s_%d_insp' %(str(ijob),jjob), 
            #'--time=', 
            '%s/submitter_chunk%s_%d.sh' %(out_dir, ijob, jjob), 
        ])

        print(command_sh_batch)
        tot_jobs+=1
        os.system(command_sh_batch)

print(tot_jobs)
