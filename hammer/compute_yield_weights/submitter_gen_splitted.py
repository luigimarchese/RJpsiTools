'''
Cambia il rng seed!
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMRandomNumberGeneratorService
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFastSimRandNumGen

from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()
process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")
'''

import os
from glob import glob
import ROOT

files = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/RJPsi_Bc_LHEGEN_11oct20_v3/RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_step1_*.root')
files = files[1:]

njobs = len(files)
#njobs = 1
out_dir = 'RJPsi_Bc_GEN_23May22_v7'
#out_dir = 'RJPsi_Bc_GEN_14jan21_v2'
events_per_job = 40000
#events_per_job = 100

template_cfg = "RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_TEMPLATE_cfg.py"
template_fileout = "RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_TEMPLATE.root"

tot_jobs = 0
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
            
for ijob in range(njobs):

    
        
    #for each line in the input file
    lhe_file = ROOT.TFile.Open(files[ijob],"r")
    nevents = lhe_file.Get("Events").GetEntries()
    #print("Number of events:",nevents)
    lhe_file.Close()
    jjobs = nevents/events_per_job
    if jjobs<= int(jjobs)+1:
        jjobs = int(jjobs)
    else:
        jjobs =int(jjobs)+1
        
    #print("Number of sub jobs for the file",jjobs)
    
    for jjob in range(jjobs):
        
        #input file
        fin = open(template_cfg, "rt")
        tmp_cfg = template_cfg.replace('TEMPLATE', 'chunk%d_%d' %(ijob,jjob))
        tmp_fileout = template_fileout.replace('TEMPLATE', '%d_%d'%(ijob,jjob))
        #output file to write the result to
        fout = open("%s/%s" %(out_dir, tmp_cfg), "wt")
        
        skip_events = jjob* events_per_job
        #print("skipped_events",skip_events)
        for line in fin:
            #read replace the string and write to output file
            if   'HOOK_FILE_IN'    in line: fout.write(line.replace('HOOK_FILE_IN'   , files[ijob]))
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
            'cmsRun {cfg}',
            'xrdcp /scratch/friti/{scratch_dir}/{fout} root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/friti/{se_dir}/{fout}',
            'rm /scratch/friti/{scratch_dir}/{fout}',
            '',
        ]).format(
            dir           = '/'.join([os.getcwd(), out_dir]), 
            scratch_dir   = out_dir, 
            cfg           = tmp_cfg, 
            se_dir        = out_dir,
            fout          = tmp_fileout
        )
        
        with open("%s/submitter_chunk%d_%d.sh" %(out_dir, ijob,jjob), "wt") as flauncher: 
            flauncher.write(to_write)
    
        command_sh_batch = ' '.join([
            'sbatch', 
            '-p long', 
            '--account=t3', 
            '-o %s/logs/chunk%d_%d.log' %(out_dir, ijob, jjob),
            '-e %s/errs/chunk%d_%d.err' %(out_dir, ijob, jjob), 
            '--job-name=gen_split' , 
            '--time=6:00:00', 
            '%s/submitter_chunk%d_%d.sh' %(out_dir, ijob, jjob), 
        ])

        print(command_sh_batch)
        tot_jobs +=1
        os.system(command_sh_batch)

print("Total jobs",tot_jobs)

    
