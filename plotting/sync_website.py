#Script to sync a specific folder with lxplus, to be able to see it on the website.

import os

#ATTENTION! NO final / after the plot folder name!
plots_folder = '23Mar2021_16h28m25s'

os.system("cp htaccess_forwebsite.txt plots_ul/"+plots_folder+"/.htaccess")

#modify htaccess for subfolders
os.system("cp htaccess_forwebsite.txt plots_ul/"+plots_folder+"/png/.htaccess")
f = open("plots_ul/"+plots_folder+"/png/.htaccess","r")
list_of_lines = f.readlines()
list_of_lines[42] = "DirectoryIndex ../../index.php\n"
f_w = open("plots_ul/"+plots_folder+"/png/.htaccess","w")
f_w.writelines(list_of_lines)
f_w.close()
os.system("cp plots_ul/"+plots_folder+"/png/.htaccess plots_ul/"+plots_folder+"/pdf/.")
os.system("cp plots_ul/"+plots_folder+"/png/.htaccess plots_ul/"+plots_folder+"/fail_region/.")
os.system("cp plots_ul/"+plots_folder+"/png/.htaccess plots_ul/"+plots_folder+"/datacards/.")

#modify htaccess for subsubfolders
os.system("cp htaccess_forwebsite.txt plots_ul/"+plots_folder+"/png/lin/.htaccess")
f = open("plots_ul/"+plots_folder+"/png/lin/.htaccess","r")
list_of_lines = f.readlines()
list_of_lines[42] = "DirectoryIndex ../../../index.php\n"
f_w = open("plots_ul/"+plots_folder+"/png/lin/.htaccess","w")
f_w.writelines(list_of_lines)
f_w.close()
os.system("cp plots_ul/"+plots_folder+"/png/lin/.htaccess plots_ul/"+plots_folder+"/png/log/.")
os.system("cp plots_ul/"+plots_folder+"/png/lin/.htaccess plots_ul/"+plots_folder+"/pdf/lin/.")
os.system("cp plots_ul/"+plots_folder+"/png/lin/.htaccess plots_ul/"+plots_folder+"/pdf/log/.")
os.system("cp plots_ul/"+plots_folder+"/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/png/.")
os.system("cp plots_ul/"+plots_folder+"/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/pdf/.")

#modify htaccess for subsubsubfolders
os.system("cp htaccess_forwebsite.txt plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess")
f = open("plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess","r")
list_of_lines = f.readlines()
list_of_lines[42] = "DirectoryIndex ../../../../index.php\n"
f_w = open("plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess","w")
f_w.writelines(list_of_lines)
f_w.close()
os.system("cp plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/png/log/.")
os.system("cp plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/png/lin/.")
os.system("cp plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/pdf/log/.")
os.system("cp plots_ul/"+plots_folder+"/fail_region/png/lin/.htaccess plots_ul/"+plots_folder+"/fail_region/pdf/lin/.")


os.system("rsync -azP plots_ul/"+plots_folder+" friti@lxplus.cern.ch:/afs/cern.ch/user/f/friti/eos/www/")
