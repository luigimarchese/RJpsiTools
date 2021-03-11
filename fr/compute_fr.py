import ROOT
from new_branches import to_define
from selections import preselection, pass_id
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

# preselection
# muonID

#Only need data

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)

#officialStyle(ROOT.gStyle, ROOT.TGaxis)

tree_name = 'BTo3Mu'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar05/'
data= ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged.root' %(tree_dir))

#define jpsiK_mass
for new_column, new_definition in to_define:
    if data.HasColumn(new_column):
        continue
    data = data.Define(new_column, new_definition)

#apply preselection
data = data.Filter(preselection)
his_model_pass = (ROOT.RDF.TH1DModel('jpsiK_mass_pass'                , '', 25,      5.,     5.45), 'J/#psiK mass (GeV)'                                               , 0)
his_model_total = (ROOT.RDF.TH1DModel('jpsiK_mass_total'                , '', 25,      5.,     5.45), 'J/#psiK mass (GeV)'                                               , 0)
models = [his_model_pass,his_model_total]

his_pass = data.Filter(pass_id).Histo1D(his_model_pass[0],"jpsiK_mass")
his_total = data.Histo1D(his_model_total[0],"jpsiK_mass")

c1 = ROOT.TCanvas("c1","",700, 700)
c1.Draw()
c1.cd()
integrals = []
funcs = []
for histo,param in zip([his_pass,his_total],[[5.28,0.05],[5.27,0.05]]):

    func= ROOT.TF1("func","gaus(0) +pol2(3)")
    func.SetParameter(1,param[0])
    func.SetParameter(2,param[1])
    #    histo.SetTitle( '; Events / bin;' + var.xlabel+var.unit+';Counts')
    fit_result=histo.Fit(func,"S")
    fit_gaus=ROOT.TF1("gaus","gaus",5,5.45)
    fit_gaus.SetLineColor(ROOT.kRed)
    fit_gaus.SetParameter(0,fit_result.Parameter(0))
    fit_gaus.SetParameter(1,fit_result.Parameter(1))
    fit_gaus.SetParameter(2,fit_result.Parameter(2))
    #legend
    
    histo.GetXaxis().SetTitle(his_model_pass[1])
    histo.GetYaxis().SetTitle('events')

    leg = ROOT.TLegend(0.65,0.70,0.98,0.86)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(histo.GetPtr(), "data", "lp")
    leg.AddEntry(fit_gaus, "gaus fit", "l")
    c1.SetTicks(True)
    c1.SetBottomMargin(2)
    c1.SetLeftMargin(0.15)
    c1.SetRightMargin(0.15)

    histo.Draw("e")

    leg.Draw("SAME")

    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '    Preliminary', lumi_13TeV = '')
    fit_gaus.Draw("lsame")

    c1.Modified()
    c1.Update()
    c1.SaveAs('gaus_fit_'+histo.GetPtr().GetName()+'.png')
    integrals.append(histo.Integral())
    funcs.append(fit_gaus)
    

print("Fake Rate = ", integrals[0]/integrals[1])

c2= ROOT.TCanvas()
funcs[0].SetMaximum(2.*max(funcs[0].GetMaximum(),funcs[1].GetMaximum()))
funcs[0].SetMinimum(0.)
funcs[0].Draw("l")
funcs[1].Draw("lsame")
c2.SaveAs("comp.png")
fr = funcs[0].Integral(0,10)/funcs[1].Integral(0,10)
print("Fake Rate = ", funcs[0].Integral(0,10)/funcs[1].Integral(0,10))
f_out=open("fake_rate.txt","w+")
f_out.write("fr = "+str(fr))
f_out.write("\n")
f_out.write("\n")
f_out.write("preselection "+ preselection)
f_out.write("\n")
f_out.write("\n")
f_out.write("passId "+ pass_id)
f_out.close()
