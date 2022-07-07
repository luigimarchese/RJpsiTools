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
#njobs = 2
out_dir = 'RJPsi_Bc_GEN_23May22_v7'
#out_dir = 'RJPsi_Bc_GEN_14jan21_v2'
events_per_job = -1
#events_per_job = 100

template_cfg = "RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_TEMPLATE_cfg.py"
template_fileout = "RJpsi-BcToXToJpsiMuMu-RunIISummer19UL18GEN_TEMPLATE.root"

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

    tmp_cfg = template_cfg.replace('TEMPLATE', 'chunk%d' %ijob)
    tmp_fileout = template_fileout.replace('TEMPLATE', '%d'%ijob)
    
    #input file
    fin = open(template_cfg, "rt")
    #output file to write the result to
    fout = open("%s/%s" %(out_dir, tmp_cfg), "wt")
    #for each line in the input file
    #lhe_file = ROOT.TFile.Open(files[ijob],"r")
    #nevents = lhe_file.Get("Events").GetEntries()
    #print(nevents)

        
    for line in fin:
        #read replace the string and write to output file
        if   'HOOK_FILE_IN'    in line: fout.write(line.replace('HOOK_FILE_IN'   , files[ijob]))
        elif 'HOOK_MAX_EVENTS' in line: fout.write(line.replace('HOOK_MAX_EVENTS', '%d' %events_per_job))
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

    with open("%s/submitter_chunk%d.sh" %(out_dir, ijob), "wt") as flauncher: 
        flauncher.write(to_write)
    
    command_sh_batch = ' '.join([
        'sbatch', 
        '-p long', 
        '--account=t3', 
        '-o %s/logs/chunk%d.log' %(out_dir, ijob),
        '-e %s/errs/chunk%d.err' %(out_dir, ijob), 
        '--job-name=genbc_1' , 
        '--time=2-00:00:00', 
        '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)

    
    
