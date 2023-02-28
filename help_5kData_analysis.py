#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:09:41 2021

@author: annika
"""


import pickle
import pandas as pd
import re
import networkx as nx
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from scipy.stats import zscore, ttest_ind, ttest_1samp, f_oneway, pearsonr
from sklearn.impute import SimpleImputer, KNNImputer
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def create_proteome_uniform_names(path_proteome):
    # protein data always come with different IDs etc. So we have to give them uniform names.
    # Also: qualtiy controls are in the data. We mark them and create one df without them and one containing them
    
    proteome = pd.read_csv(path_proteome,index_col=0) # contains quality controls
    
    measured_proteins_ORF = list(proteome.index)            
    KO_proteins_ENSG_raw = list(proteome.columns)
    
    KO_proteins_ENSG_lessraw = [element[element.find('_ko_')+len('_ko_'):element.rfind('_')] for element in KO_proteins_ENSG_raw]
    KO_proteins_ENSG = [element[:element.rfind('_')] for element in KO_proteins_ENSG_lessraw]
    proteome.columns = KO_proteins_ENSG
       
    # converted the above lists with gprofiler
    df_uni2ensg=pd.read_csv('output/measured_proteins_uniprot2ENSG.csv', index_col=0)  # index: Uniprot. ['converted_alias']: ENSG
    df_ensg2uni=pd.read_csv('output/KO_proteins_ENSG2uniprot.csv', index_col=0)  # index: ENSG. ['converted_alias']: Uniprot
    
    s = df_uni2ensg['converted_alias']
    t = df_uni2ensg['converted_alias'].transpose()
    df_ensg2uniprot = pd.concat([s,t], axis=0)
    df_ensg2uniprot = df_ensg2uniprot[~df_ensg2uniprot.index.duplicated(keep='first')]
    
    for old_name in proteome.index:
        proteome.rename(index={old_name: df_ensg2uniprot.loc[old_name]}, inplace=True)
        
    quality_controls = [column for column in proteome.columns if ('_qc_' in column) or ('HIS' in column)]
    no_quality_controls = [column for column in proteome.columns if not column in quality_controls]
    no_quality_controls = [element for element in no_quality_controls if not ('CONT' in element) or ('DNG' in element)]
    proteome_QC_only = proteome[quality_controls]
    proteome_QC_only = proteome_QC_only[~proteome_QC_only.index.duplicated(keep='first')]
    proteome_QC_only = proteome_QC_only.loc[:,~proteome_QC_only.columns.duplicated(keep='first')]
    
    proteome_no_QC = proteome[no_quality_controls]
    proteome_no_QC = proteome_no_QC[~proteome_no_QC.index.duplicated(keep='first')]
    proteome_no_QC = proteome_no_QC.loc[:,~proteome_no_QC.columns.duplicated(keep='first')]
    
    proteome = proteome.transpose()
    proteome_no_QC = proteome_no_QC.transpose()
    proteome_QC_only = proteome_QC_only.transpose()
    
    return proteome, proteome_no_QC, proteome_QC_only

def create_proteome_uniform_names_withImpute(path_proteome, path_nr_de_proteins, de_proteins, whichImputer):
    # Same as function above but now we impute missing values. 
    
    proteome = pd.read_csv(path_proteome,index_col=0) # contains quality controls
    
    measured_proteins_ORF = list(proteome.index)
            
    KO_proteins_ENSG_raw = list(proteome.columns)
    KO_proteins_ENSG_lessraw = [element[element.find('_ko_')+len('_ko_'):element.rfind('_')] for element in KO_proteins_ENSG_raw]
    KO_proteins_ENSG = [element[:element.rfind('_')] for element in KO_proteins_ENSG_lessraw]
    proteome.columns = KO_proteins_ENSG
       
    # converted the above lists with gprofiler
    df_uni2ensg=pd.read_csv('../5kDataAnalysis/output/measured_proteins_uniprot2ENSG.csv', index_col=0)  # index: Uniprot. ['converted_alias']: ENSG
    df_ensg2uni=pd.read_csv('../5kDataAnalysis/output/KO_proteins_ENSG2uniprot.csv', index_col=0)  # index: ENSG. ['converted_alias']: Uniprot
    
    s = df_uni2ensg['converted_alias']
    t = df_uni2ensg['converted_alias'].transpose()
    df_ensg2uniprot = pd.concat([s,t], axis=0)
    df_ensg2uniprot = df_ensg2uniprot[~df_ensg2uniprot.index.duplicated(keep='first')]
    
    for old_name in proteome.index:
        proteome.rename(index={old_name: df_ensg2uniprot.loc[old_name]}, inplace=True)
    for old_name in de_proteins.index:
        de_proteins.rename(index={old_name: df_ensg2uniprot.loc[old_name]}, inplace=True)
        
    quality_controls = [column for column in proteome.columns if ('_qc_' in column) or ('HIS' in column)]
    no_quality_controls = [column for column in proteome.columns if not column in quality_controls]
    no_quality_controls = [element for element in no_quality_controls if not ('CONT' in element) or ('DNG' in element)]
    proteome_QC_only = proteome[quality_controls]
    proteome_QC_only = proteome_QC_only[~proteome_QC_only.index.duplicated(keep='first')]
    proteome_QC_only = proteome_QC_only.loc[:,~proteome_QC_only.columns.duplicated(keep='first')]
    
    proteome_no_QC = proteome[no_quality_controls]
    proteome_no_QC = proteome_no_QC[~proteome_no_QC.index.duplicated(keep='first')]
    proteome_no_QC = proteome_no_QC.loc[:,~proteome_no_QC.columns.duplicated(keep='first')]
    
    proteome = proteome.transpose()
    proteome_no_QC = proteome_no_QC.transpose()
    proteome_QC_only = proteome_QC_only.transpose()
    
    if whichImputer=='Simple':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        proteome_no_QC = pd.DataFrame(data=imp.fit(proteome_no_QC).transform(proteome_no_QC), columns=proteome_no_QC.columns, index=proteome_no_QC.index)
    if whichImputer=='KNN':
        imp =KNNImputer(n_neighbors=5, weights="uniform")
        proteome_no_QC = pd.DataFrame(data=imp.fit(proteome_no_QC).transform(proteome_no_QC), columns=proteome_no_QC.columns, index=proteome_no_QC.index)
    nr_de_proteins_in_KOs = pd.read_csv(path_nr_de_proteins,index_col=0)
    nr_de_proteins_in_KOs_raw = list(nr_de_proteins_in_KOs.index)
    nr_de_proteins_in_KOs_lessraw = [element[element.find('_ko_')+len('_ko_'):element.rfind('_')] for element in nr_de_proteins_in_KOs_raw]
    nr_de_proteins_in_KOs_names = [element[:element.rfind('_')] for element in nr_de_proteins_in_KOs_lessraw]
    nr_de_proteins_in_KOs.index = nr_de_proteins_in_KOs_names
    nr_de_proteins_in_KOs = nr_de_proteins_in_KOs.loc[~nr_de_proteins_in_KOs.index.duplicated(keep='first')]
    
    return proteome, proteome_no_QC, proteome_QC_only, nr_de_proteins_in_KOs, de_proteins

def compute_de_proteins(path_de_proteins):
    # read in differentially expressed proteins computed by Christoph
    
    
    de_proteins_original = pd.read_csv(path_de_proteins,index_col=0)
    proteins = set(list(de_proteins_original.index))
    
    de_proteins = pd.DataFrame(index=proteins, columns=['count'])
    for protein in proteins:
        part_df = de_proteins_original[de_proteins_original['p.adjust']<=0.05].loc[protein]
        de_proteins['count'][protein] = len(part_df)
    
    return de_proteins


def do_the_corr4PPI_degree(network_name, dict_zscores, dict_network_paths):
    #  computes the correlation between degree in the network and SD in the samples of each protein
        
    network = nx.read_weighted_edgelist(dict_network_paths[network_name])
    proteins_network = list(network.nodes)
    proteins_measured = list(dict_zscores.keys())
    overlap = [gene for gene in proteins_network if gene in proteins_measured]  # proteins that are both in the network and in the sample
    
    # each pair of protein & gene is an element. 
    x_values_list =[]  # SD of protein
    y_values_list =[]  # degree  of protein
    for gene in overlap:
        x_values_list.append(abs(dict_zscores[gene]))
        y_values_list.append(network.degree[gene])
            
    x_values = np.array(x_values_list).astype(np.float)  # convert to float
    y_values = np.array(y_values_list).astype(np.float)
    
    # where the figures are stored
    path_out_fig = 'output/LinRegVariabilityvsDegreePPI/CorrelationDegreevsSD_' + network_name + '.png'
    if os.path.exists(path_out_fig):
        os.remove(path_out_fig)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)  # compute correlatio
    # plot figure
    plt.title("Correlation Degree vs. SD")
    plt.xlabel('SD')
    plt.ylabel('Degree of genes in yeast PPI ' + network_name)
    plt.plot(x_values, y_values, 'o', label='original data')
    plt.plot(x_values, intercept + slope * x_values, 'r', label='fitted line')
    plt.savefig(path_out_fig, dpi=600, bbox_inches='tight')
    plt.close()

    # histogram
    cutoff = 0#1.5
    x_values_cutoff = x_values.copy()
    y_values_cutoff = y_values.copy()
    
    path_out_fig_density = 'output/LinRegVariabilityvsDegreePPI/CorrelationDegreevsSD_density_' + network_name 
    max_degree = 0
    min_degree = 2
    
    if network_name == 'Bader':
        max_degree = 250
    if network_name == 'BioGrid':
        max_degree = 1304
    if network_name == 'STRING':
        max_degree = 1300     
    
    #*************************************************************************
    # For total numbers:
    #*************************************************************************
    
    # x_values: zscores
    # y_values: degrees
    
    plt.title("Distribution of variability vs degree in PPI")
    #  need defined number of bins for the histogram, otherwise too crowded
    bins = np.round(np.arange(min(x_values),max(x_values),.1), 1)
    number_curves = 10
    color=(cm.rainbow(np.linspace(0,1,number_curves)))
    
    deg_distribution = np.linspace(min_degree, max_degree, 11)  # number of bins 11, start from min_degree to max_degree
    degrees = []  # each entry is a list with a two entries, which define the range of degrees within that bin
    for index in range(10):
        degree_start = int(deg_distribution[index])
        degree_stop  = int(deg_distribution[index+1])
        degrees.append([degree_start, degree_stop-1])
    
    f,ax = plt.subplots(1,1,sharey=True, facecolor='w')
    for plot_degrees in degrees: 
        start_degree = plot_degrees[0]
        end_degree = plot_degrees[1]        
    
        [x_toplot, y_toplot] = np.histogram(x_values_cutoff[(y_values_cutoff>=start_degree)*(y_values_cutoff<=end_degree)], bins=bins)
        y_toplot = np.delete(y_toplot, -1)
        y_toplot = y_toplot[y_toplot>0]
        x_toplot = x_toplot[x_toplot>0]
        total_draws = sum(x_toplot)
        drawn_successes = x_toplot
        
        number_current_distance = sum((y_values_cutoff>=start_degree)*(y_values_cutoff<=end_degree))
        font_size =10
        if len(set(plot_degrees)) ==1:
            ax.plot(y_toplot, x_toplot, 'o', label='degree ' + str(start_degree) + ' (' + str(number_current_distance) + ')', alpha=0.5)

        else:
            ax.plot(y_toplot, x_toplot, 'o', label='degree ' + str(start_degree) + ' to ' + str(end_degree) + ' (' + str(number_current_distance) + ')', alpha=0.5)

    ax.set_xlabel('SD')
    ax.set_ylabel('Number of degree of genes in yeast PPI')
    
    ax.legend(loc='upper right')
    plt.savefig(path_out_fig_density + '_totalNRs.png', dpi=600, bbox_inches='tight')
    plt.close()

    #*************************************************************************
    # For ratios:
    #*************************************************************************
    
    # x_values: zscores
    # y_values: degrees
    
    plt.title("Distribution of variability vs degree in PPI")
    
    bins = np.round(np.arange(min(x_values),max(x_values),.1), 1)
    number_curves = 10+7
    color=(cm.rainbow(np.linspace(0,1,number_curves)))
    
      
    # degrees = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10, 35]]
    f,ax = plt.subplots(1,1,sharey=True, facecolor='w')
    for plot_degrees in degrees: 
        start_degree = plot_degrees[0]
        end_degree = plot_degrees[1]        
    
        [x_toplot, y_toplot] = np.histogram(x_values_cutoff[(y_values_cutoff>=start_degree)*(y_values_cutoff<=end_degree)], bins=bins)
        y_toplot = np.delete(y_toplot, -1)
        y_toplot = y_toplot[y_toplot>0]
        x_toplot = x_toplot[x_toplot>0]
        total_draws = sum(x_toplot)
        drawn_successes = x_toplot

        # normalize: 
        number_current_distance = sum((y_values_cutoff>=start_degree)*(y_values_cutoff<=end_degree))
        x_toplot = x_toplot/number_current_distance
        font_size =10
        if len(set(plot_degrees)) ==1:
            ax.plot(y_toplot, x_toplot, 'o', label='degree ' + str(start_degree) + ' (' + str(number_current_distance) + ')', alpha=0.5)
        else:
            ax.plot(y_toplot, x_toplot, 'o', label='degree ' + str(start_degree) + ' to ' + str(end_degree) + ' (' + str(number_current_distance) + ')', alpha=0.5)
    
    ax.set_xlabel('SD')
    ax.set_ylabel('Ratio of Number of degree of genes in PPI')
    ax.legend(loc='upper right')
   
 
    plt.savefig(path_out_fig_density + '_ratios.png', dpi=600, bbox_inches='tight')
    plt.close()
    
def do_the_PPI_degree_sigma_boxplot(network_name, dict_zscores, dict_zscores_KOs, dict_network_paths, number_bins):
    #  plot the SD of proteins across the samples vs degree in a PPI. 
    # we create equally sized bins based on degrees and for each set of proteins/bins a boxplot is plotted
    
    
    network = nx.read_weighted_edgelist(dict_network_paths[network_name])
    proteins_network = list(network.nodes)
    proteins_measured = list(dict_zscores.keys())
    overlap = [gene for gene in proteins_network if gene in proteins_measured]
    
    # create dataframe: 
    df = pd.DataFrame(index=['degree', 'SD'], columns=overlap) # the dataframe contains the degree and the standard deviation of the proteins
    for protein in overlap:
        df[protein]['degree'] = network.degree[protein]
        df[protein]['SD'] = abs(dict_zscores[protein])
    df = df.transpose().astype(float)
        
    out_path = 'output/boxplots/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    
    # Binning: Equal sized bins
    res, bins = pd.qcut(df['degree'], number_bins, retbins=True, duplicates='drop')
    
    for index_bin in range(len(bins)-1):
        df.loc[(df['degree'] >= int(bins[index_bin]))&(df['degree']< int(bins[index_bin+1])), 'degree_group'] = index_bin
    fig, ax = plt.subplots()
    
    title = 'PPI_' + network_name + '_degree_vs_SD_bins'
    sns_plot = sns.boxplot(x="degree_group", y="SD", data=df).set(title=title)
    plt.savefig(out_path + title + '.png')
    plt.close()  
    
   
    # create dataframe for KOs: 
    
    proteins_measured = list(dict_zscores_KOs.keys())
    overlap = [gene for gene in proteins_network if gene in proteins_measured]
    
    df = pd.DataFrame(index=['degree', 'SD'], columns=overlap) # the dataframe contains the degree and the standard deviation of the proteins
    for protein in overlap:
        df[protein]['degree'] = network.degree[protein]
        df[protein]['SD'] = abs(dict_zscores_KOs[protein])
    df = df.transpose().astype(float)
        
    out_path = 'output/boxplots/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Binning: Equal sized bins
    res, bins = pd.qcut(df['degree'], number_bins, retbins=True, duplicates='drop')
    
    for index_bin in range(len(bins)-1):
        df.loc[(df['degree'] >= int(bins[index_bin]))&(df['degree']< int(bins[index_bin+1])), 'degree_group'] = index_bin
    fig, ax = plt.subplots()
    
    title = 'PPI_' + network_name + '_degree_vs_SD_bins_KOs'
    sns_plot = sns.boxplot(x="degree_group", y="SD", data=df).set(title=title)
    plt.savefig(out_path + title + '.png')
    plt.close()  
    
    
   
    
def boxplot_responsive_vs_nonresponsive(network_name, dict_zscores, dict_network_paths, responisve_KOs, figure_name):
    #  given: KnockOuts that are responisve and non-responsive.
    #  we plot their degree in a given network and compare the responsive to the non-resp. (boxplot)
    
    network = nx.read_weighted_edgelist(dict_network_paths[network_name])
    proteins_network = list(network.nodes)
    proteins_measured = list(dict_zscores.keys())
    
    overlap = list(set(set(proteins_measured)&set(proteins_network)))
    
    df = pd.DataFrame(index=['degree', 'responsive'], columns=overlap) # the dataframe contains the degree and the standard deviation of the proteins
    
    
    for protein in overlap:
        df[protein]['degree'] = network.degree[protein]
        if protein in responisve_KOs:
            df[protein]['responsive'] = 1
        else:
            df[protein]['responsive'] = 0
        
    df = df.transpose().astype(float)
    
    pvalue = ttest_ind(df[df['responsive'] == 0]['degree'],df[df['responsive'] == 1]['degree'], equal_var = True)[1]
    out_path = 'output/boxplots/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fig, ax = plt.subplots()
    title = 'PPI_' + network_name + '_' + figure_name + '_degree_vs_responsive_pvalue_' + str(round(pvalue,4))
    sns_plot = sns.boxplot(x="responsive", y="degree", data=df).set(title=title)
    plt.savefig(out_path + title + '.png')
    plt.close()
    
    # log2 
    df['degree'] = np.log2(df['degree'])
    pvalue = ttest_ind(df[df['responsive'] == 0]['degree'],df[df['responsive'] == 1]['degree'], equal_var = True)[1]
    out_path = 'output/boxplots/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fig, ax = plt.subplots()
    title = 'PPI_' + network_name + '_' + figure_name + '_log2_degree_vs_responsive_pvalue_' + str(round(pvalue,4))
    sns_plot = sns.boxplot(x="responsive", y="degree", data=df).set(title=title)
    plt.savefig(out_path + title + '.png')
    plt.close()
    return df

def get_proteins_from_files(path_proteins):
    proteins = []
    file_nodes = open(path_proteins, 'r')  # open file where the nodes are in
    for line in file_nodes:  # go through each row (contains one node)
        li = line.strip()  # prepare
        if not li.startswith('#'):  # check if comment
            content = li.split('\t')  # I think this is not necessary but I left it in anyways ...
            proteins.append(content[1].upper())  # node is first entry (there shouldn't be a second one, but safety first)
    return proteins

def compute_DE_proteins(proteome_no_QC, proteome_QC_only, zscore_threshold):
    # based on given zscores, check if a protein is differentially expressed (DE)
    
    # proteome_QC_only: df; indexes are the sample names, columns are the measured proteins
    # proteome_no_QC: df; indexes are the samples, thus the 5k KOs, columns are the measured proteins
    
    mean_protein_quantities = proteome_QC_only.mean()
    zscores_protein = (proteome_no_QC - proteome_QC_only.mean())/proteome_QC_only.std()
    
    dict_samples2DEproteins = {}
    for sample in zscores_protein.index:
        dict_samples2DEproteins[sample] = [protein for protein in zscores_protein.loc[sample].index if abs(zscores_protein.loc[sample][protein]) >= zscore_threshold]
    
    dict_DEprotein2sample = {}
    for protein in zscores_protein.columns:
        dict_DEprotein2sample[protein] = [sample for sample in zscores_protein[protein].index if abs(zscores_protein.loc[sample][protein]) >= zscore_threshold]
    
    return mean_protein_quantities, zscores_protein, dict_samples2DEproteins, dict_DEprotein2sample
    

def read_in_de_proteins(path_de_proteins):
    # read in proteins and measurements and convert the protein IDs into different IDs
    
    df_de_proteins = pd.read_csv(path_de_proteins,index_col=0) # contains quality controls
    measured_proteins_ORF = list(df_de_proteins['Protein.Group'])  

    KO_proteins_ENSG = list(df_de_proteins['orf'])
       
    # converted the above lists with gprofiler
    df_uni2ensg=pd.read_csv('../5kDataAnalysis/output/measured_proteins_uniprot2ENSG.csv', index_col=0)  # index: Uniprot. ['converted_alias']: ENSG
    df_ensg2uni=pd.read_csv('../5kDataAnalysis/output/KO_proteins_ENSG2uniprot.csv', index_col=0)  # index: ENSG. ['converted_alias']: Uniprot
    
    s = df_uni2ensg['converted_alias']
    t = df_uni2ensg['converted_alias'].transpose()
    
    df_ensg2uni.rename({'converted_alias':'duh'}, axis=1, inplace=True)
    df_ensg2uni['converted_alias'] = df_ensg2uni.index
    df_ensg2uni.rename({'duh':'initial_alias'}, axis=1, inplace=True)
    
    u = df_ensg2uni['converted_alias']
    
    df_ensg2uniprot = pd.concat([s,u], axis=0)
    df_ensg2uniprot = df_ensg2uniprot[~df_ensg2uniprot.index.duplicated(keep='first')]
    dict_ensg2uniprot = df_ensg2uniprot.to_dict()
    df_de_proteins.index = list((range(len(df_de_proteins.index))))
    df_de_proteins_wide = df_de_proteins.pivot_table(columns='Protein.Group',index='orf', values='p.adjust')#, aggfunc='mean')
    df_de_proteins_wide.rename(dict_ensg2uniprot, axis=0, inplace=True) 
    df_de_proteins_wide.rename(dict_ensg2uniprot, axis=1, inplace=True) 
    df_de_proteins_wide = df_de_proteins_wide.astype(float)
    
    return df_de_proteins_wide
    