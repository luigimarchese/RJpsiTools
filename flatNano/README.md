# Flattener of NanoAODs

# Environment
All the scripts (except for `files_path_writer.py`), work with the conda environment and Hammer installation described [here](https://github.com/friti/RJpsiTools/tree/main/hammer).
Remember to activate the environment
`conda activate hammer3p8`

# The scripts
1. `Resonant_rjpsi.py` -> flattens the nanoAOD. It takes as argument the txt files that contain the paths for the nanoAOD (see 2021Jan folder). The options depend on the kind of sample you want to process. 
2. `submitter.py` -> it submits the jobs into the bash of PSI Tier3 (need the access). It's possible to change the number of files per job, the path of the txt files, the dataset.
3. `check_files.py` -> it checks that the submitted jobs finished without errors and it prints which output files are missing
4. `same_resubmitter.py` -> resubmit the jobs that failed, without changing the number of files per jobs
5. `split_jobs_resubmitter.py` -> resubmit the jobs failed, you can choose another number of files per job (< of the first one and such that old%new=0). This can not be used for BcToX dataset.
***
6. `files_path_writer.py` -> if you sent CRAB jobs to produce the nanoAOD, you can use this script to print the file paths into a txt file,that you can use to run the flattener. This script need the CMSSW environment!


