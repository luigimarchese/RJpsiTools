'''
Cambia il rng seed!
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMRandomNumberGeneratorService
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFastSimRandNumGen

In the end of the cfg file!
from IOMC.RandomEngine.RandomServiceHelper import  RandomNumberServiceHelper
randHelper =  RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randHelper.populate()
process.RandomNumberGeneratorService.saveFileName =  cms.untracked.string("RandomEngineState.log")


'''
import os
from glob import glob
from files_HbToJPsiMuMu_3MuFilter_old import files


nfiles = len(files)
out_dir = 'Rjpsi-HbToJpsiMuMu-3MuFilter_AOD_inspector_oldversion_scale1_v2'
files_per_job = 1
njobs = int(nfiles/files_per_job)
print(njobs," will be submitted")

template_inspector = "inspector_TEMPLATE.py"
template_fileout = "RJpsi-HbToJpsiMuMu-3MuFilter_AOD_inspector_oldversion_scale1_v2_TEMPLATE.root"

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
os.system('cp files_HbToJPsiMuMu_3MuFilter_old.py ' + out_dir + '/.')
os.system('cp -r GeneratorInterface ' + out_dir + '/.')

for ijob in range(njobs):

    tmp_inspector = template_inspector.replace('TEMPLATE', 'chunk%d' %ijob)
    tmp_fileout = template_fileout.replace('TEMPLATE', '%d'%ijob)
    
    #input file
    fin = open(template_inspector, "rt")
    #output file to write the result to
    fout = open("%s/%s" %(out_dir, tmp_inspector), "wt")
    #for each line in the input file
    for line in fin:
        #read replace the string and write to output file
        if   'HOOK_FILE_IN'    in line: fout.write(line.replace('HOOK_INPUT'   , files[ijob]))
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
        'python {insp} --verbose 0 --files_per_job {files_per_job} --jobid {jobid}',
        'xrdcp /scratch/friti/{scratch_dir}/{fout} root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/friti/{se_dir}/{fout}',
        'rm /scratch/friti/{scratch_dir}/{fout}',
        'echo {fout} Saved!',
        '',
    ]).format(
        dir           = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir   = out_dir, 
        insp           = tmp_inspector, 
        files_per_job = files_per_job,
        se_dir        = out_dir,
        jobid         = ijob,
        fout          = tmp_fileout
        )

    with open("%s/submitter_chunk%d.sh" %(out_dir, ijob), "wt") as flauncher: 
        flauncher.write(to_write)
    
    command_sh_batch = ' '.join([
        'sbatch', 
        #'-p testnew', 
        '--account=t3', 
        '-o %s/logs/chunk%d.log' %(out_dir, ijob),
        '-e %s/errs/chunk%d.err' %(out_dir, ijob), 
        '--job-name=%s_insp' %str(ijob), 
        #'--time=60', 
        '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
    ])

    os.system(command_sh_batch)
