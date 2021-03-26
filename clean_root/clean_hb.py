from root_pandas import read_root
from root_pandas import to_root

#open file root
original_file_hb = "/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_ptmax_merged.root"

#clean from trigger and from bc
df_hb = read_root(original_file_hb,'BTo3Mu',where = 'mu1_isFromMuT & mu2_isFromMuT & mu1_isFromJpsi_MuT & mu2_isFromJpsi_MuT & k_isFromMuT & (abs(mu2_grandgrandmother_pdgId) != 421 | abs(mu1_grandgrandmother_pdgId) != 421)')

df_hb.to_root("/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_trigger_bcclean.root",key = 'BTo3Mu')

#open file root
original_file_hbmu = "/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar16/HbToJPsiMuMu3MuFilter_ptmax_merged.root"

#clean from trigger and from bc
df_hbmu = read_root(original_file_hbmu,'BTo3Mu',where = 'mu1_isFromMuT & mu2_isFromMuT & mu1_isFromJpsi_MuT & mu2_isFromJpsi_MuT & k_isFromMuT & (abs(mu2_grandgrandmother_pdgId) != 421 | abs(mu1_grandgrandmother_pdgId) != 421)')

df_hbmu.to_root("/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar16/HbToJPsiMuMu3MuFilter_trigger_bcclean.root",key = 'BTo3Mu')
