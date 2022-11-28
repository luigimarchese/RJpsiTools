#system
import os
import copy
from datetime import datetime
import random
import time
import sys
import multiprocessing as mp

# computation libraries
import ROOT
import pandas as pd
import numpy as np
from array import array
import pickle
import math 
from bokeh.palettes import viridis, all_palettes
from keras.models import load_model
from ROOT import * #Added for the dimuon combinatorial background

# cms libs
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

# personal libs
from histos import histos as histos_lm
from new_branches import to_define
from samples import weights, titles, colours
#from selections import prepreselection, triggerselection, preselection, preselection_mc, preselectionLSB, preselectionRSB, preselectionSRForSB, pass_id, fail_id
from selections import preselection_general, triggerselection, prepreselection, preselection_mc, pass_id, fail_id
from create_datacard_v3 import create_datacard_ch1, create_datacard_ch2, create_datacard_ch3, create_datacard_ch4, create_datacard_ch1_onlypass, create_datacard_ch3_onlypass
#from plot_shape_nuisances_v4 import plot_shape_nuisances # In this import there is an option ROOT.gROOT.SetBatch() which will prevent to draw histos and canvas live (h.Draw()) even if you allow python to keep waiting for an input (with input() at the end of the .py).If you need to Draw live histos you need to comment this import

#ROOT.ROOT.EnableImplicitMT()
data_w = 5
alpha_w = 5
mc_w = 5

def get_DiMuonBkgNorm(label):
    
    tree_name = 'BTo3Mu'
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'
    tree_dir_low = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2022Nov22/'

    dataframe = {}
    dataframe["SR"] = ROOT.RDataFrame(tree_name,'%s/data_nopresel_withpresel_v2_withnn_withidiso_v3.root'%(tree_dir))
    dataframe["SBs"] = ROOT.RDataFrame(tree_name,{'%s/datalowmass_ptmax_merged_v5.root'%(tree_dir_low), '%s/data_nopresel_withpresel_v2_withnn_withidiso_v3.root'%(tree_dir)})
    regions = list(dataframe.keys())

    print("======================================")
    print("==== Dimuon Combinatorial Bkg Norm====")
    print("======================================")
    print("regions: ", regions)

    sanitycheck                           = True
    JpsimassSRloose                       = {}
    JpsimassShape                         = {}

    ######################
    ##### Defintions #####
    ######################
   
    SR_min  = 2.96
    SR_max  = 3.23
    Bkgshape_min   = 2.4
    Bkgshape_max   = 2.8

    ### Get the relevant histos and information from the DataFrames  ###
    #preprepreselection = ' & '.join([prepreselection]) # no trigger no eta
    ### Get the histo with the full invariant-mass distribution to extract the Dimuon shape ### 
    #filterSBsShape = ' & '.join([prepreselection,'Bmass<6.3', pass_id])    

    # no trigger no eta
    filterSBsShape = ' & '.join([prepreselection])
    JpsimassShape["SBs"] = dataframe["SBs"].Filter(filterSBsShape).Histo1D(("mJpsiSBShape","mJpsiSBShape;  m_{#mu#mu} [GeV]; Events/0.01 GeV", 200, 2, 4), "jpsi_mass")
    HJpsimassSB = JpsimassShape["SBs"].GetValue()
    os.system('mkdir -p plots_ul/'+label+'/dimuon/norm/')
    if(sanitycheck):
        SBMasscanvas = TCanvas("SBMassc", "SBMassc")
        SBMasscanvas.cd()
        JpsimassShape["SBs"].Draw("pe")
        SBMasscanvas.Print('plots_ul/'+label+'/dimuon/norm/SBMasscanvas.png')
    
    
    ### Get the histo in the SR with the SR loose selection to perform the fit to get the Dimuon normalization ###

    filterSRloose = ' & '.join([prepreselection, triggerselection])
    JpsimassSRloose["SR"] = dataframe["SR"].Filter(filterSRloose).Histo1D(("mJpsiSRloose","mJpsiSRloose;  m_{#mu#mu} [GeV]; Events/0.01 GeV", 200, 2, 4), "jpsi_mass")
    HJpsimassSRloose = JpsimassSRloose["SR"].GetValue()
    #SRdataset = ROOT.RooDataSet('SRdataset', 'SRdataset', tree_name, filterSR)
    if(sanitycheck):
        SRMasscanvas = TCanvas("SRMassc", "SRMassc")
        SRMasscanvas.cd()
        JpsimassSRloose["SR"].Draw("pe")
        SRMasscanvas.Print('plots_ul/'+label+'/dimuon/norm/SRMasscanvas.png')

    #################################### 
    ####### Dimuon Normalization #######
    ####################################

    # Shape defintion # 
                    
    mass                   = ROOT.RooRealVar     ("mass",           "mass",                  SR_min,        SR_max                               )
    massSB                 = ROOT.RooRealVar     ("massSB",         "massSB",                Bkgshape_min,  Bkgshape_max                         )
    
    mass.setRange  ('SR',        SR_min,        SR_max       )
    massSB.setRange('Bkgshape',  Bkgshape_min,  Bkgshape_max )
    
    #####    Signal    #####
    MassJpsi               = ROOT.RooRealVar     ("MassJpsi",       "MassJpsi",     3.0969                                                       )
    scale                  = ROOT.RooRealVar     ("scale",          "scale",        1.,      0.,   2.                                            )
    sigma                  = ROOT.RooRealVar     ("sigma",          "sigma",        0.03,    0.,   0.5                                           )
    ResSigma               = ROOT.RooRealVar     ("ResSigma",       "ResSigma",     1.5,     0.,   3.                                            )
    alphaCB                = ROOT.RooRealVar     ("alphaCB",        "alphaCB",      1.5,     0.,   10.                                           )
    nCB                    = ROOT.RooRealVar     ("nCB",             "nCB",         1.5,     0.,   100.                                          )
    fraGauss               = ROOT.RooRealVar     ("fraGauss",       "fraGauss",     0.1,     0.,   0.35                                          )
    
    MeanMassJpsi           = ROOT.RooFormulaVar  ("MeanMassJpsi",   "MassJpsi*scale", ROOT.RooArgList(MassJpsi, scale))
    sigmaGauss             = ROOT.RooFormulaVar  ("sigmaGauss",     "sigma*ResSigma", ROOT.RooArgList(sigma, ResSigma))
    
    #####   Background   #####
    bkgSlope               = ROOT.RooRealVar     ("bkgSlope",        "bkgSlope",    -4.,     -20., 20.                                           )
    
    ##### Fit Normalizations #####
    NSgl                   = ROOT.RooRealVar     ("NSgl",            "NSgl",        5000,     0.,  500000.                                       )
    NBkg                   = ROOT.RooRealVar     ("NBkg",            "NBkg",        500,      0.,    4000.                                       )
    NBkgSB                 = ROOT.RooRealVar     ("NBkgSB",          "NBkgSB",      5000,     0.,  100000.                                       )
    
    #####################
    #####   PDFs    #####
    #####################
    CBall                  = ROOT.RooCBShape     ("CBall",            "CBall",      mass, MeanMassJpsi, sigma, alphaCB, nCB                      )
    Gauss                  = ROOT.RooGaussian    ("Gauss",            "Gauss",      mass, MeanMassJpsi, sigmaGauss                               )
    SigPDF                 = ROOT.RooAddPdf      ("SigPDF",           "SigPDF",     ROOT.RooArgList(Gauss, CBall), ROOT.RooArgList(fraGauss)     )
    
    Expo                   = ROOT.RooExponential ("Expo",             "Expo",       mass,   bkgSlope                                             )
    SBExpo                 = ROOT.RooExponential ("SBExpo",           "SBExpo",     massSB, bkgSlope                                             )
    
    shapes                 = ROOT.RooArgList     (SigPDF, Expo)
    yields                 = ROOT.RooArgList     (NSgl,   NBkg)
    
    PDFSB                  = ROOT.RooAddPdf      ("PDFSB",        "PDFSB",          ROOT.RooArgList(SBExpo), ROOT.RooArgList(NBkgSB)             )
    CompletePDF            = ROOT.RooAddPdf      ("CompletePDF",  "CompletePDF",    ROOT.RooArgList(shapes), ROOT.RooArgList(yields)             )

    # Fit to the invariant mass in the SB to extract ths background shape #
    SBdataset   = ROOT.RooDataHist("SBdataset", "SBdataset", ROOT.RooArgList(massSB), HJpsimassSB)
    framemassSB = massSB.frame(ROOT.RooFit.Name(""), ROOT.RooFit.Title(""), ROOT.RooFit.Bins(200))
    SBdataset.plotOn(framemassSB,ROOT.RooFit.Binning(200, 2, 4),ROOT.RooFit.MarkerSize(1.5))
    if(sanitycheck):
        PDFSB.fitTo(SBdataset,ROOT.RooFit.Save())
    else:
        PDFSB.fitTo(SBdataset,ROOT.RooFit.Save(),RooFit.PrintLevel(-1),RooFit.PrintEvalErrors(-1))
    
    PDFSB.plotOn(framemassSB)
    if(sanitycheck):
        JpsiMassSBCanvas = TCanvas("FitShapeJpsiMassSB", "FitShapeJpsiMassSB")
        JpsiMassSBCanvas.cd()
        framemassSB.Draw()
        JpsiMassSBCanvas.Print('plots_ul/'+label+'/dimuon/norm/FitShapeJpsiMassSB.png')
    
    BkgShapetoFix = bkgSlope.getVal()                                       
    if(sanitycheck):
        print("Shape slope for the dimuon from SB fit:", BkgShapetoFix)
    
                 
    # Fit to the invariant mass in the SR (defined with the loose analysis selections) #
    SRdataset = ROOT.RooDataHist("SRdataset", "SRdataset", mass, HJpsimassSRloose)
    framemass = mass.frame(ROOT.RooFit.Title(""), ROOT.RooFit.Bins(200))    
    SRdataset.plotOn(framemass,ROOT.RooFit.Name("data"),ROOT.RooFit.Binning(200, 2, 4))
    bkgSlope.setConstant(ROOT.kTRUE)
    bkgSlope.setVal(BkgShapetoFix)
    if(sanitycheck):
        CompletePDF.fitTo(SRdataset)
    else:
        CompletePDF.fitTo(SRdataset,ROOT.RooFit.Save(),RooFit.PrintLevel(-1),RooFit.PrintEvalErrors(-1))
    CompletePDF.plotOn(framemass)
    CompletePDF.plotOn(framemass,ROOT.RooFit.Components("Expo"),ROOT.RooFit.LineColor(ROOT.kGreen),ROOT.RooFit.FillColor(ROOT.kGreen),ROOT.RooFit.DrawOption("F"),ROOT.RooFit.MoveToBack())
    CompletePDF.plotOn(framemass,ROOT.RooFit.Components("CBall"),ROOT.RooFit.LineColor(ROOT.kGray),ROOT.RooFit.FillColor(ROOT.kGray),ROOT.RooFit.DrawOption("F"),ROOT.RooFit.MoveToBack())
    CompletePDF.plotOn(framemass,ROOT.RooFit.Components("Gauss"),ROOT.RooFit.LineColor(ROOT.kRed),ROOT.RooFit.LineStyle(ROOT.kDashed));
    framemass.GetXaxis().SetTitleSize(0.1)
    framemass.GetXaxis().SetLabelSize(0.05)
    framemass.GetYaxis().SetTitleOffset(0.85)
    framemass.GetYaxis().SetTitleSize(0.05)
    framemass.GetYaxis().SetNdivisions(505)
    framemass.SetYTitle("Events/0.01 GeV")
    framemass.SetTitle(" ")
    
    #################
    ## Pulls study ##
    #################
    SRdatasetPulls = framemass.pullHist()
    framemassPulls = mass.frame(ROOT.RooFit.Title(""))
    framemassPulls.addPlotable(SRdatasetPulls,"P") 
    framemassPulls.GetXaxis().SetTitleSize(0.08)
    framemassPulls.GetXaxis().SetLabelSize(0.05)
    framemassPulls.GetYaxis().SetTitleOffset(0.5)
    framemassPulls.GetYaxis().CenterTitle(1)
    framemassPulls.GetYaxis().SetTitle("#Delta/#sigma")
    framemassPulls.GetXaxis().SetTitle("m_{#mu#mu} [GeV]")
    framemassPulls.GetYaxis().SetTitleSize(0.08)
    framemassPulls.GetYaxis().SetLabelSize(0.05)
    framemassPulls.GetYaxis().SetNdivisions(505)
    #§framemassPulls.GetYaxis().SetRangeUser(-10.52, 10.52)
    framemassPulls.Draw();
    line = TLine(2.,0,4.,0)
    line.SetLineColor(kRed)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    
    ##############
    ## Plotting ##
    ##############
    JpsiMassSRCanvas = TCanvas("JpsiMassSR", "JpsiMassSR", 0, 0, 700, 700)
    JpsiMassSRCanvas.SetTicks()
    JpsiMassSRCanvas.SetTopMargin(0.015);
    JpsiMassSRCanvas.SetRightMargin(0.020);
    JpsiMassSRCanvas.SetBottomMargin(0.15);
    JpsiMassSRCanvas.SetLeftMargin(0.12);
    JpsiMassSRCanvas.cd()
    Plot = TPad("Plot", "Plot", 0, 0.4, 1, 1)
    Pulls = TPad("Pulls", "Pulls", 0, 0, 1, 0.4)
    Plot.SetRightMargin(0.02)
    Plot.SetLeftMargin(0.16)
    Plot.SetTopMargin(0.02)
    Plot.SetBottomMargin(0.001)
    Pulls.SetRightMargin(0.02)
    Pulls.SetTopMargin(0)
    Pulls.SetBottomMargin(0.45)
    Pulls.SetLeftMargin(0.16)
    Plot.SetTicks()
    Pulls.SetTicks()
    Plot.Draw()
    Pulls.Draw()

    
    cms = TLatex()
    cms.SetTextSize(0.07)
    
    leg = TLegend(0.18,0.7,0.4,0.2)
    leg.SetFillColor(ROOT.kWhite)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.05)
    leg.SetTextFont(42)
    leg.AddEntry(framemass.getObject(2),"Data","PLE")
    leg.AddEntry(framemass.getObject(3),"Fit pdf","L")
    leg.AddEntry(framemass.getObject(0),"Crystall Ball","FL")
    leg.AddEntry(framemass.getObject(4),"Gaussian","L")
    leg.AddEntry(framemass.getObject(1),"Background","FL")
    
    if(sanitycheck):
        Plot.cd()
        framemass.Draw("")
        cms.DrawLatexNDC(0.2, 0.85, "#it{CMS} #bf{Internal}")
        leg.Draw()
        Pulls.cd()
        framemassPulls.Draw("")
        #line.Draw("same")
        #framemass.Draw()
        #Jpsimass["SR"].Draw("pe")
        JpsiMassSRCanvas.Print('plots_ul/'+label+'/dimuon/norm/FitJpsiMassSR.png')
        #CompletePDF.Draw("SAME")

    Normalization = NBkg.getVal()/JpsimassSRloose["SR"].Integral()
    # The final NBkg = NBkg_SRloose * N_entires_SR/N_entries_SRloose. This function returns the Number of background events/N_entries_SRloose. The function get_DiMuonBkg() will later scale it by the missing N_entires_SR
    if(sanitycheck):
        print("NBkg: ", NBkg.getVal(), "Jpsimass[SRloose].Integral(): ",  JpsimassSRloose["SR"].Integral(), "NormalizationSRloose: ", Normalization)
    print("DiMuon Normalization done")

    return Normalization
    

def get_DiMuonBkg(NormSRloose, label):
    # NormSRloose is the normalization from the SRloose, obtained by calling the function above get_DiMuonBkgNorm()
    # selection is the category selection which will be used to derive the specific DiMuon shape (Q2 or jpsivtx_log10_lxy_sig) in that category
    # var_index is the variable_index used to understand which shape variable we are interested in (Q2 or jpsivtx_log10_lxy_sig, but also several others in case in future we will need them for the BDT)
    # isfail is a flag included to specify if we need to apply the neural network weights for the fakes. When isfail = 1 we don't apply these weights, it's the flat_fakerate channel (ch2_flat); when isfail = 0 we do apply these weights, it's the not flat_fakerate (ch2)
    # label is the date-format name for the directory where these shapes will be saved
    # channel is the channel number for the different categories 

    tree_name = 'BTo3Mu'
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'
    tree_dir_low = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2022Nov22/'
        
    dataframe = {}
    #dataframe["SR"] = ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged_fakerate.root'%(tree_dir))
    #dataframe["SR"] = ROOT.RDataFrame(tree_name,'%s/data_fakerate_only_iso.root'%(tree_dir))
    #dataframe["ResonantTrg"] = ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged_fakerate.root'%(tree_dir)) 
    #dataframe["NonResonantTrg"] = ROOT.RDataFrame(tree_name,'%s/datalowmass_ptmax_merged_fakerate.root'%(tree_dir))
    #dataframe["SBs"] = ROOT.RDataFrame(tree_name,{'%s/datalowmass_ptmax_merged_fakerate_2.root'%(tree_dir), '%s/data_ptmax_merged_fakerate.root'%(tree_dir)})
    dataframe["SR"] = ROOT.RDataFrame(tree_name,'%s/data_nopresel_withpresel_v2_withnn_withidiso_v3.root'%(tree_dir))
    dataframe["SBs"] = ROOT.RDataFrame(tree_name,{'%s/datalowmass_withfakes.root'%(tree_dir_low), '%s/data_nopresel_withpresel_v2_withnn_withidiso_v3.root'%(tree_dir)})
    regions = list(dataframe.keys())

    
    print("========================================")
    print("==== Dimuon Combinatorial Bkg Shape ====")
    print("========================================")
    print("regions: ", regions)


    sanitycheck                           = True
    
    #hists                                = {}
    Q2hist                                = {}
    Q2_extrap_hist                        = {}
    m_miss_extrap_hist                    = {}
    pt_var_extrap_hist                    = {}
    pt_miss_vec_extrap_hist               = {}
    pt_miss_scal_extrap_hist              = {}
    m_miss_extrap_hist                    = {}
    jpsivtx_log10_lxy_sig_estrap_hist     = {}
    JpsimassShape                         = {}
    Jpsimass                              = {}
    JpsimassSB                            = {}
    JpsimassLSB                           = {}
    JpsimassSR                            = {}
    Q2LSBcanvas                           = {}
    Q2_extrapcanvas                       = {}
    DimuonShape                           = {}
    
    
    ######################
    ##### Defintions #####
    ######################
    
    LSB_min = 2.89
    LSB_max = 3.01
    RSB_min = 3.18
    RSB_max = 3.32
   
    # if in the fail region, cut also on the NN weights that are not under control

    #define new columns
    for s in ["SR", "SBs"]:
        for new_column, new_definition in to_define: 
            if dataframe[s].HasColumn(new_column):
                continue       
            dataframe[s] = dataframe[s].Define(new_column, new_definition)
        
    ### Get the relevant histos and information from the DataFrames  ###
    
    ### LSB ###
    #filterLSB = ' & '.join([preselectionLSB, pass_id])
    filterLSB = ' & '.join([prepreselection, triggerselection, 'jpsi_mass>%s'%LSB_min, 'jpsi_mass<%s'%LSB_max])
    #dataframe["SBs"].Display({"k_softMvaId"},120).Print()
    Q2hist["SBs"] = dataframe["SBs"].Filter(filterLSB).Histo1D(("Q2LSB","Q2LSB;  q^{2} [GeV^{2}]; Events/0.5 GeV",24,0,10.5),"Q_sq")

    ### SR for this category ###
    filterSR = ' & '.join([prepreselection, triggerselection])
    JpsimassSR["SR"] = dataframe["SR"].Filter(filterSR).Histo1D(("mJpsiSR","mJpsiSR;  m_{#mu#mu} [GeV]; Events/0.01 GeV", 200, 2, 4), "jpsi_mass")
    HJpsimassSR = JpsimassSR["SR"].GetValue()
              
    ### Get the scale factor to extrapolate the LSB to the SR ###
    JpsimassLSB["SBs"] = dataframe["SBs"].Filter(filterLSB).Histo1D(("mJpsiLSB","mJpsiLSB;  m_{#mu#mu} [GeV]; Events/0.01 GeV", 200, 2, 4), "jpsi_mass")
    HJpsimassLSB = JpsimassLSB["SBs"].GetValue()
    Jpsi_scale = 3.0969/HJpsimassLSB.GetMean()
    dataframe["SBs"] = dataframe["SBs"].Filter(filterLSB).Define("Jpsi_scale", "{}".format(Jpsi_scale))
    if(sanitycheck):
        print("Mean For Jpsi_scale: ", HJpsimassLSB.GetMean())
        Q2LSBcanvas = TCanvas("Q2LSBcan", "Q2LSBcan")
        Q2LSBcanvas.cd()
        Q2hist["SBs"].Draw("pe")
        Q2LSBcanvas.Print('plots_ul/'+label+'/dimuon/Q2LSBcan.png') 

    ############################# 
    ####### Dimuon Shape  #######
    #############################
                
    ROOT.gInterpreter.Declare(
        """
        using Vec_t = const ROOT::VecOps::RVec<float>;
        float SB_extrap(float B_pt_reco, int variable, float scale,
        float pt1, float eta1, float phi1, float m1, 
        float pt2, float eta2, float phi2, float m2, 
        float pt3, float eta3, float phi3, float m3) {
        float Bc_MASS_PDG = 6.275;
        //cout<<pt1<<" "<<eta1<<" "<<phi1<<" "<<m1<<" "<<endl;
        //cout<<pt2<<" "<<eta2<<" "<<phi2<<" "<<m2<<" "<<endl;
        //cout<<pt3<<" "<<eta3<<" "<<phi3<<" "<<m3<<" "<<endl;
        //cout<<scale<<endl;
        TLorentzVector mu1_p4, mu2_p4, mu3_p4, B_coll_p4, Jpsi_p4_extrap;
        mu1_p4.SetPtEtaPhiM(pt1, eta1, phi1, m1);
        mu2_p4.SetPtEtaPhiM(pt1, eta2, phi2, m2);
        mu3_p4.SetPtEtaPhiM(pt3, eta3, phi3, m3);
        B_coll_p4.SetPtEtaPhiM(B_pt_reco, (mu1_p4 + mu2_p4 + mu3_p4).Eta(), (mu1_p4 + mu2_p4 + mu3_p4).Phi(), Bc_MASS_PDG);
        Jpsi_p4_extrap.SetPtEtaPhiM((mu1_p4 + mu2_p4).Pt(), (mu1_p4 + mu2_p4).Eta(), (mu1_p4 + mu2_p4).Phi(), (mu1_p4 + mu2_p4).M()*scale);
      
        Float_t Q_sq = (B_coll_p4 - Jpsi_p4_extrap)*(B_coll_p4 - Jpsi_p4_extrap);
        Float_t m_miss_sq = (B_coll_p4 - Jpsi_p4_extrap - mu3_p4)*(B_coll_p4 - Jpsi_p4_extrap - mu3_p4);
        Float_t pt_var = (Jpsi_p4_extrap.Pt() - mu3_p4.Pt());
        Float_t pt_miss_vec = ((B_coll_p4 - mu3_p4 - Jpsi_p4_extrap).Pt());
        Float_t pt_miss_scal = (B_coll_p4.Pt() - mu3_p4.Pt() - Jpsi_p4_extrap.Pt());

        Float_t RetVar;
        if (variable == 0)      RetVar = Q_sq;
        else if (variable == 1) RetVar = m_miss_sq;
        else if (variable == 2) RetVar = pt_var;
        else if (variable == 3) RetVar = pt_miss_vec;
        else if (variable == 4) RetVar = pt_miss_scal;

        return RetVar;
        }
        """)


    dataframe["SBs"] = dataframe["SBs"].Filter(filterLSB).Define("Q_sq_extrap", "SB_extrap(Bpt_reco, 0, Jpsi_scale, mu1pt, mu1eta, mu1phi, mu1mass, mu2pt, mu2eta, mu2phi, mu2mass, kpt, keta, kphi, kmass)")
    Q2_extrap_hist["SBs"] = dataframe["SBs"].Filter(filterLSB).Histo1D(("Q_sqLSB_extrap","Q_sqLSB_extrap;  q^{2} [GeV^{2}];",20,0,10),"Q_sq_extrap")
    DimuonShape = Q2_extrap_hist["SBs"]
        
    if(sanitycheck):
        Q2_extrapcanvas = TCanvas("Q2_extrapcanvasc", "Q2_extrapcanvasc")
        Q2_extrapcanvas.cd()
        DimuonShape.Draw("pe")
        Q2_extrapcanvas.Print('plots_ul/'+label+'/dimuon/Q2_extrapcanvas.png')

    FinalNorm = NormSRloose*JpsimassSR["SR"].Integral()/DimuonShape.Integral()
    # The final NBkg = NBkg_SRloose * N_entires_SR/N_entries_SRloose. The function get_DiMuonBkgNorm returns the Number of background NormSRloose = events/N_entries_SRloose. The function get_DiMuonBkg() is now scaling it by the missing N_entires_SR. The shape histo must be normalized to unity in order to apply the final normalization, so we need to scale NormSRloose by N_entires_SR/DimuonShape.Integral()
    DimuonShape.Scale(FinalNorm)
    if(sanitycheck):
        print("Dimuon Normalization from SRloose: ", NormSRloose)
        print("Jpsimass[SR].Integral(): ",  JpsimassSR["SR"].Integral(), "Jpsimass[SR].Entries(): ",  JpsimassSR["SR"].GetEntries(), "DimuonShape.Integral() : ", DimuonShape.Integral())
        print("Final DiMuon number of events for this category = : ", NormSRloose*JpsimassSR["SR"].Integral())
        DiMuonShapeCanvas = TCanvas("DiMuonShapeCanvas", "DiMuonShapeCanvas", 0, 0, 700, 700)
        DiMuonShapeCanvas.cd()
        DimuonShape.GetValue().Draw("pe")
        DiMuonShapeCanvas.Print('plots_ul/'+label+'/dimuon/NormalizedDiMuonShape.png')
    print("DiMuon done")
    return FinalNorm
                    
