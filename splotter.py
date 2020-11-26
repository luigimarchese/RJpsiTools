import ROOT

ff = ROOT.TFile.Open('jpsi_pi_splots.root')
tree = ff.Get('tree')

ths = ROOT.THStack('ths', '')

h_all = ROOT.TH1F('h_all', '', 20, 0, 1)
h_sig = ROOT.TH1F('h_sig', '', 20, 0, 1)
h_bkg = ROOT.TH1F('h_bkg', '', 20, 0, 1)

tree.Draw('Bsvprob >> h_all', ''       , 'hist')
tree.Draw('Bsvprob >> h_sig', 'nsig_sw', 'hist')
tree.Draw('Bsvprob >> h_bkg', 'nbkg_sw', 'hist')

h_sig.SetLineColor(ROOT.kRed)
h_bkg.SetLineColor(ROOT.kBlue)

h_sig.SetFillColor(ROOT.kRed)
h_bkg.SetFillColor(ROOT.kBlue)

h_all.SetMarkerStyle(8)

ths.Add(h_sig)
ths.Add(h_bkg)

ths.Draw('hist')
h_all.Draw('EP same')
ROOT.gPad.SaveAs('svprob_splot.pdf')
