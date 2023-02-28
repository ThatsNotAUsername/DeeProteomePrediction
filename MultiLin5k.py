# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:07:43 2020

@author: roehla

Using proteomics data from different KnockOuts we identify dependencies between 
protein levels, where a protein level of one protein can be dependent on the 
levels of several other proteins. 

"""

import pandas as pd
import networkx as nx
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../HelpFunctions')
import help_5kData_analysis
import ML_LinearRegression
import os
import itertools
import time

def read_in_network(path2nw):
    file_network = open(path2nw, 'rb')
    network = nx.read_weighted_edgelist(file_network)  # read in graph from file
    file_network.close()
    return network

# feature selection
def select_features(X_train, y_train, X_test, number_best_features):
	# configure to select all features
    fs = SelectKBest(score_func=f_regression, k=number_best_features)
	# learn relationship from training data
    fs.fit(X_train, y_train)
        
	# transform train input data
    X_train_fs = fs.transform(X_train)
	# transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def do_regression(X,y, number_best_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)  # to produce different split, use different random_state integer
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_predict = model.predict(X_test)
    # evaluate predictions
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, number_best_features)
    mask = fs.get_support()
    best_features = []
    feature_names = list(X_train.columns)
    
    for bool, feature in zip(mask, feature_names):
        if bool:
            best_features.append(feature)
    
    
    model.fit(X_train_fs, y_train)
    # evaluate the model
    y_predict_fs = model.predict(X_test_fs)
    # evaluate predictions
    rmse_fs = mean_squared_error(y_test, y_predict_fs, squared=False)
    # print('RMSE fs selected features: %.3f' % rmse_fs)
    return rmse, fs, best_features


# n_features_to_select = 5

# Networks:
# HumanNet: /home/annika/Charite/Networks/HumanNet/*.tsv: entrez IDs
#   -SC-GT: Genetic interactions in S. cerevisiae (Genes: 2,671, Links 44,222)
#   -SC-HT: low-throughput PPI in S. cerevisiae (2,326, 39,536)
#   -SC-LC: Literature-curated PPI in S. cerevisiae (2,580, 23,974)
#   -SC-TS: Protein 3D structure based interactions in S. cerevisiae	(857, 6,005)

# Biogrid /home/annika/Charite/Networks/BioGrid/BIOGRID-ORGANISM-3.5.187.tab3/*.txt
#   - BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.187_EntrezOnly : the biogrid using entrez IDs


# network by Bader and Hogue /home/annika/Charite/Networks/MergedNetworks/*.txt
#   - AllNWsTogether_uniqueEdges: ORF IDs, all networks merged

# network by Schwikowski:
#   - /home/annika/Charite/Networks/SchwikowskiEtAl/schwikowski_ORFOnly.txt


# path_protein_data = '/home/annika/Charite/Charite/FiveKYeast/Data/Data_log2FC.csv'  # Names of the rows and columns are cleaned by Oliver and values are given as Fold Change, not measured. Protein names are ORF and columns are ORF of the KO
#path_protein_data_imp = main_home_folder + '/Charite/FiveKYeast/Data/Data_ko_imputed.csv'  # Names of the rows and columns are cleaned by Oliver and values are given as Fold Change, not measured. Protein names are ORF and columns are ORF of the KO



forNetwork = False  # TRUE: Use a given network
forAllProteins = False  # TRUE: Try it for all proteins (very costly)
do_PCA = False  # do PCA in order to find some features/proteins

for_high_abundance = False  # use only high abundand proteins
for_low_abundance = False  #  use only low abundand proteins
for_tenth_abundance = True #  use only 10% of highest abundand proteins

for_high_abundance = True


# Maximal percentage of NaN entries
max_nan = .4
# KNN imputer neighbors
KNN = 5
'''
data_georg = pickle.load(open('../../Data/Georgs/ProteomeHD_v1_1_cleaned.pkl', 'rb'))
Genes_to_drop_nan = [index for index in data_georg.index if np.sum(pd.isnull(data_georg.loc[index].values))/len(data_georg.columns) > max_nan]
data_georg.drop(labels=Genes_to_drop_nan, axis=0, inplace=True)
# data_cancer = np.log2(data_cancer)
imputer = KNNImputer(n_neighbors=KNN, weights='uniform', metric='nan_euclidean')
data_georg[:] = (imputer.fit_transform(data_georg))

dict_prot2gene_georg = pickle.load(open('../../Data/Georgs/dict_prot2GeneName.pkl', 'rb'))
old_index_georg = list(data_georg.index)
new_index_georg = [dict_prot2gene_georg[protein] for protein in old_index_georg]
data_georg.index = new_index_georg
columns_georg = list(data_georg.columns)
all_data_log = pd.DataFrame(index=list(set(new_index_georg)), columns=columns_georg)
number_columns = len(columns_georg)

for protein in new_index_georg:
        temp = data_georg.loc[protein]
        if isinstance(temp, pd.DataFrame):  # the protein is in the dataframe more than once
        # go through all rows. Chekc for all entries if nan. If all nan, leave nan, else replace with mean
            for index_col in range(number_columns):
                sum_numbers = sum(temp.iloc[:, index_col])
                all_data_log.loc[protein,columns_georg[index_col]] = sum_numbers/len(list(temp.index))
        else:
            all_data_log.loc[protein] = temp
'''

# read in the imputed data by Christoph
'''
path_5k_imputed = 'C:/Charite/Data/5k/data_long_maxlfq_lib0225_modimpute_20210416_format.RDat'
result = pyreadr.read_r(path_5k_imputed) # also works for Rds
'''

# ****************************************************************************
# ******************* different parameters will be tried out ******************************
# ****************************************************************************


network_names = ['BioGrid'] #, 'STRING']  # which networls should be used
network_name='BioGrid'
alphas = [0.001, 0.01, 0.05, 0.1, 0.4, 0.7, 1, 8, 100, 500, 800] #  list(np.linspace(0.1, 800, 5))

which_imputed = [False]#, 'Simple', 'KNN']#, 'Simple', False]   # hwo the missing values are imputed
how_many_proteins = 10  #  how many proteins are used as features
from_number_proteins = 150



columns = ['Model', ' alpha', 'l1Ratio', 'Imputation', 'NW', 'ProteinName', 'r2scoreTest', 'r2scoreTrain', 'RunTime', 'NrNeighbors', 'RMSE_Test', 'RMSE_Train']
df_all_results = pd.DataFrame(columns=columns)

#  which models should be tried out
do_Lasso = True
do_Ridge = True
do_ElasticNet = True
do_SVM = True
do_Forest = True

counter=from_number_proteins
# use_self_imputed = False

for use_self_imputed in which_imputed:
    
    # where is the data
    path_nr_de_proteins = '../../Data/5k/20210423/5k_nr_de_proteins_filename_20210421.csv'
    path_proteome = '../../Data/5k/20210423/5k_quant_wide_20210416.csv'  # 
    path_proteome_notImputed = '../../Data/5k/20210423/5k_quant_wide_20210416_noimpute.csv'  # 
    path_de_proteins = '../../Data/5k/20210423/5k_de_proteins_filename_20210421.csv'
    de_proteins_old_names = pd.DataFrame(index=[], columns=[])
    
    # read in proteome
    if not use_self_imputed:
        proteome, proteome_no_QC, proteome_QC_only, nr_de_proteins_in_KOs, de_proteins = help_5kData_analysis.create_proteome_uniform_names(path_proteome, path_nr_de_proteins, de_proteins_old_names)
    else:
        proteome, proteome_no_QC, proteome_QC_only, nr_de_proteins_in_KOs, de_proteins = help_5kData_analysis.create_proteome_uniform_names_withImpute(path_proteome_notImputed, path_nr_de_proteins, de_proteins_old_names, use_self_imputed)
    

    # we need the non-imputed to exclude non-measured proteins from the prediction
    proteome_notImputed, proteome_no_QC_notImputed, proteome_QC_only_notImputed, nr_de_proteins_in_KOs_notImputed, de_proteins_notImputed = help_5kData_analysis.create_proteome_uniform_names(path_proteome_notImputed, path_nr_de_proteins, de_proteins_old_names)
    
    measured_proteins = list(proteome.columns)  # protein names
    KO_proteins = list(proteome.index)  # sample names,; names of the proteins which were knocked out
    proteome_no_QC_log = np.log2(proteome_no_QC)  # log2 transform the proteome
    
    if forNetwork:  # if we use first neighbors in network as features
        for network_name in network_names:  
            path_df_save = 'output/ML/df_all_results_' + network_name + '.pkl'  # where the results are stored
            pickle.dump(df_all_results, open(path_df_save,'wb'))
    
            path_figures_NWs = 'output/ML/' + network_name + '/VisualizeLinReg/'  # where the figures are stored
            if not os.path.exists(path_figures_NWs):
                os.makedirs(path_figures_NWs)
                
            # Network: read in and check proteins
            dict_network_paths = pickle.load(open('../../Networks/dict_network_paths.pkl', 'rb'))  # path where the network can be found
            network = nx.read_weighted_edgelist(dict_network_paths[network_name])  # read in network 
            prots_network = list(network.nodes)  # proteins which are in the network 
            overlap_measured = [protein for protein in measured_proteins if protein in network]  # overlap of proteins in network and measured
            overlap_KOs = [protein for protein in KO_proteins if protein in network]  # overlap of proteins in network and samples/KOs
            
            # results for different models are stored in dictionaries
            dict_proteins = {}  # for each protein the best model and parameters are stored. 
            dict_to_return_Lasso = {}
            dict_to_return_SimpleRegression = {}
            dict_to_return_ElasticNet = {}
            dict_to_return_Ridge = {}
            dict_to_return_SVM = {}
            
            for index_protein, protein in enumerate(overlap_measured[from_number_proteins:how_many_proteins]):
                print('\n##############################################################')
                print('Proteins so far: ', counter)
                print('##############################################################\n')
                counter+=1
                dict_proteins[protein] = {}
                dict_to_return_Lasso[protein] = {}
                dict_to_return_SimpleRegression[protein] = {}
                dict_to_return_ElasticNet[protein] = {}
                dict_to_return_Ridge[protein] = {}
                dict_to_return_SVM[protein] = {}
                first_neighbors = [neighbor for neighbor in list(network.neighbors(protein)) if neighbor in overlap_measured]  # these are the only proteins we accept as features
                
                # only proceed if we actually have neighbors in the network for the protein
                if protein in first_neighbors:
                    first_neighbors.remove(protein)
                if len(first_neighbors):
                    X = proteome_no_QC_log[first_neighbors].transpose()
                    X = X.apply(pd.to_numeric)
                    y = proteome_no_QC_log[protein]
                    y = y.apply(pd.to_numeric)
                    
                    # remove samples, where the protein is not measured:
                    not_measured_samples = list(proteome_no_QC_notImputed[protein].index[proteome_no_QC_notImputed[protein].isnull()])
                    X.drop(not_measured_samples, axis=1, inplace=True)
                    y.drop(not_measured_samples, inplace=True)
                    
                    if len(X.columns):
                        # y = np.asarray(y).reshape(-1,1)
                        # X = np.asarray(X).reshape(-1,1)
                        number_best_features = int(len(X.columns)/5)
                        if not number_best_features:
                            number_best_features =1
                        r2test = float('nan')
                        model ='model'
                        alpha_to_use = float('nan')
                        l1_to_use = float('nan')
                        n_split = min(5, len(first_neighbors))
                        
                        # proceed if enough samples
                        # try different alphas and models
                        if n_split>=2:
                            for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                                if do_Lasso:
                                    start_time = time.time()
                                    dict_to_return_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha)
                                    print('Time for Lasso in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Lasso[protein][alpha]['r2_test'], dict_to_return_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_Lasso[protein][alpha]['RMSE_test'], dict_to_return_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))

                                if do_Ridge:
                                    start_time = time.time()
                                    dict_to_return_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha)
                                    print('Time for Ridge in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Ridge[protein][alpha]['r2_test'], dict_to_return_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_Ridge[protein][alpha]['RMSE_test'], dict_to_return_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                if do_ElasticNet:
                                    dict_to_return_ElasticNet[protein][alpha] = {}
                                    for l1 in [0.1, 0.5, .9]:
                                        start_time = time.time()
                                        dict_to_return_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1)
                                        print('Time for ElasticNet in NW: ', (time.time() - start_time))
                                        df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, network_name, protein, dict_to_return_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_test'], dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_train']]], columns=columns))
                                        pickle.dump(df_all_results, open(path_df_save,'wb'))
                                        
                                if do_SVM:
                                    start_time = time.time() 
                                    dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                                    print('Time for default SVM in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                                    
                                    start_time = time.time()
                                    dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                                    df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                    print('Time for linear SVM in NW: ', (time.time() - start_time))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                                    
                                    start_time = time.time()
                                    dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                                    print('Time for poly SVM in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                                    
                        if do_Forest: 
                            if n_split>=2:
                                a = [[1,45]]#a = [[10,50,100,200], [1,0.5,0.1], [None, 3,10,20]], [1,3,10]] # parameters for n_estimators=100, max_features=1.0, max_depth=None, min_samples_leaf=1
                                # a = [[120], [0.5], [3], [3]] # best parameters for both datasets
                                b = list(itertools.product(*a))  # all combinations of the parameters
                                for index_param, combi_params in enumerate(b): 
                                    n_estimators = 100
                                    max_features = 'auto'
                                    max_depth = 3
                                    min_samples_leaf = combi_params[0]
                                    class_weight='balanced'
                                    start_time = time.time()
                                    dict_RandomForest = ML_LinearRegression.Forest_regression(X.transpose(),y, n_split=n_split, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                                    print('Time for RandomForest in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['Forest', min_samples_leaf, max_depth, use_self_imputed, network_name, protein, dict_RandomForest['r2_test'], dict_RandomForest['r2_train'],  (time.time() - start_time), len(first_neighbors), dict_RandomForest['RMSE_test'], dict_RandomForest['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                            
            print('\n##############################################################')
            print('done with the network part')
            print('##############################################################\n')
            pickle.dump(df_all_results, open(path_df_save,'wb'))
    
    # only use high abundant proteins as features
    if for_high_abundance:
        network_name = 'high_abundance'
        path_df_save = 'output/ML/df_all_results_highest_abundance.pkl'
        df_abundance = pd.DataFrame(index=measured_proteins, columns=['MedianAbundance'])
        for protein in measured_proteins:
            df_abundance['MedianAbundance'][protein] = proteome_no_QC_log[protein].median()
        df_abundance_ordered = df_abundance.sort_values(by='MedianAbundance', ascending=False)
        ten_percent_highest_abundant = list(df_abundance_ordered.index)[:int(len(list(df_abundance_ordered.index))/10)]
        
        # store results for different proteins and models and parameters in dictionaries
        dict_proteins = {}  # for each protein the best model and parameters are stored. 
        dict_to_return_Lasso = {}
        dict_to_return_SimpleRegression = {}
        dict_to_return_ElasticNet = {}
        dict_to_return_Ridge = {}
        dict_to_return_SVM = {}
        
        for index_protein, protein in enumerate(measured_proteins[from_number_proteins:how_many_proteins]):
            print('\n##############################################################')
            print('Proteins so far: ', counter)
            print('##############################################################\n')
            counter+=1
            dict_proteins[protein] = {}
            dict_to_return_Lasso[protein] = {}
            dict_to_return_SimpleRegression[protein] = {}
            dict_to_return_ElasticNet[protein] = {}
            dict_to_return_Ridge[protein] = {}
            dict_to_return_SVM[protein] = {}
            features_to_use = ten_percent_highest_abundant.copy()
            if protein in features_to_use:
                features_to_use.remove(protein)
            if len(features_to_use):
                X = proteome_no_QC_log[features_to_use].transpose()
                X = X.apply(pd.to_numeric)
                y = proteome_no_QC_log[protein]
                y = y.apply(pd.to_numeric)
                
                # remove samples, where the protein is not measured:
                not_measured_samples = list(proteome_no_QC_notImputed[protein].index[proteome_no_QC_notImputed[protein].isnull()])
                X.drop(not_measured_samples, axis=1, inplace=True)
                y.drop(not_measured_samples, inplace=True)
                
                if len(X.columns):
                    # y = np.asarray(y).reshape(-1,1)
                    # X = np.asarray(X).reshape(-1,1)
                    number_best_features = int(len(X.columns)/5)
                    if not number_best_features:
                        number_best_features =1
                    r2test = float('nan')
                    model ='model'
                    alpha_to_use = float('nan')
                    l1_to_use = float('nan')
                    n_split = min(5, len(features_to_use))
                    if n_split>=2:
                        for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                            if do_Lasso:
                                start_time = time.time()
                                dict_to_return_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha)
                                print('Time for Lasso in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Lasso[protein][alpha]['r2_test'], dict_to_return_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Lasso[protein][alpha]['RMSE_test'], dict_to_return_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            # dict_to_return_SimpleRegression[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SimpleRegression', alpha=alpha)
                            if do_Ridge:
                                start_time = time.time()
                                dict_to_return_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha)
                                print('Time for Ridge in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Ridge[protein][alpha]['r2_test'], dict_to_return_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Ridge[protein][alpha]['RMSE_test'], dict_to_return_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            
                            if do_ElasticNet:
                                dict_to_return_ElasticNet[protein][alpha] = {}
                                for l1 in [0.1, 0.5, .9]:
                                    start_time = time.time()
                                    dict_to_return_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1)
                                    print('Time for ElasticNet in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, network_name, protein, dict_to_return_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_test'], dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                            if do_SVM:
                                start_time = time.time() 
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                                print('Time for default SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                print('Time for linear SVM in NW: ', (time.time() - start_time))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                                print('Time for poly SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                # start_time = time.time()
                                # dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='sigmoid')
                                # print('Time for sigmoid SVM in NW: ', (time.time() - start_time))
                                # df_all_results = df_all_results.append(pd.DataFrame([['SVMsigmoid', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                # pickle.dump(df_all_results, open('output/ML/df_all_results_' + network_name + '.pkl','wb'))
                    if do_Forest: 
                        if n_split>=2:
                            a = [[1,45]]#a = [[10,50,100,200], [1,0.5,0.1], [None, 3,10,20]], [1,3,10]] # parameters for n_estimators=100, max_features=1.0, max_depth=None, min_samples_leaf=1
                            # a = [[120], [0.5], [3], [3]] # best parameters for both datasets
                            b = list(itertools.product(*a))  # all combinations of the parameters
                            for index_param, combi_params in enumerate(b): 
                                # n_estimators = combi_params[0]
                                n_estimators = 100
                                # max_features = combi_params[1]
                                max_features = 'auto'
                                max_depth = 3
                                # min_samples_leaf = combi_params[3]
                                min_samples_leaf = combi_params[0]
                                # param = {'n_estimators':combi_params[0], "max_features":combi_params[1], "max_depth":combi_params[2]}#, "min_samples_leaf":combi_params[3]}
                                class_weight='balanced'
                                start_time = time.time()
                                dict_RandomForest = ML_LinearRegression.Forest_regression(X.transpose(),y, n_split=n_split, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                                print('Time for RandomForest in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Forest', min_samples_leaf, max_depth, use_self_imputed, network_name, protein, dict_RandomForest['r2_test'], dict_RandomForest['r2_train'],  (time.time() - start_time), len(features_to_use), dict_RandomForest['RMSE_test'], dict_RandomForest['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                        
        print('\n##############################################################')
        print('done with the network part')
        print('##############################################################\n')
        pickle.dump(df_all_results, open(path_df_save,'wb'))
        
    # oly use low abundant proteins as features
    if for_low_abundance:
        network_name = 'low_abundance'
        path_df_save = 'output/ML/df_all_results_lowest_abundance.pkl'
        df_abundance = pd.DataFrame(index=measured_proteins, columns=['MedianAbundance'])
        for protein in measured_proteins:
            df_abundance['MedianAbundance'][protein] = proteome_no_QC_log[protein].median()
        df_abundance_ordered = df_abundance.sort_values(by='MedianAbundance', ascending=True)
        ten_percent_lowest_abundant = list(df_abundance_ordered.index)[:int(len(list(df_abundance_ordered.index))/10)]
        
        dict_proteins = {}  # for each protein the best model and parameters are stored. 
        dict_to_return_Lasso = {}
        dict_to_return_SimpleRegression = {}
        dict_to_return_ElasticNet = {}
        dict_to_return_Ridge = {}
        dict_to_return_SVM = {}
        
        for index_protein, protein in enumerate(measured_proteins[from_number_proteins:how_many_proteins]):
            print('\n##############################################################')
            print('Proteins so far: ', counter)
            print('##############################################################\n')
            counter+=1
            
            # store results for different proteins and different models in dictionaries
            dict_proteins[protein] = {}
            dict_to_return_Lasso[protein] = {}
            dict_to_return_SimpleRegression[protein] = {}
            dict_to_return_ElasticNet[protein] = {}
            dict_to_return_Ridge[protein] = {}
            dict_to_return_SVM[protein] = {}
            features_to_use = ten_percent_lowest_abundant.copy()
            if protein in features_to_use:
                features_to_use.remove(protein)
            if len(features_to_use):
                X = proteome_no_QC_log[features_to_use].transpose()
                X = X.apply(pd.to_numeric)
                y = proteome_no_QC_log[protein]
                y = y.apply(pd.to_numeric)
                
                # remove samples, where the protein is not measured:
                not_measured_samples = list(proteome_no_QC_notImputed[protein].index[proteome_no_QC_notImputed[protein].isnull()])
                X.drop(not_measured_samples, axis=1, inplace=True)
                y.drop(not_measured_samples, inplace=True)
                
                if len(X.columns):
                    # y = np.asarray(y).reshape(-1,1)
                    # X = np.asarray(X).reshape(-1,1)
                    number_best_features = int(len(X.columns)/5)
                    if not number_best_features:
                        number_best_features =1
                    r2test = float('nan')
                    model ='model'
                    alpha_to_use = float('nan')
                    l1_to_use = float('nan')
                    n_split = min(5, len(features_to_use))
                    if n_split>=2:
                        for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                            if do_Lasso:
                                start_time = time.time()
                                dict_to_return_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha)
                                print('Time for Lasso in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Lasso[protein][alpha]['r2_test'], dict_to_return_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Lasso[protein][alpha]['RMSE_test'], dict_to_return_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            # dict_to_return_SimpleRegression[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SimpleRegression', alpha=alpha)
                            if do_Ridge:
                                start_time = time.time()
                                dict_to_return_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha)
                                print('Time for Ridge in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Ridge[protein][alpha]['r2_test'], dict_to_return_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Ridge[protein][alpha]['RMSE_test'], dict_to_return_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            
                            if do_ElasticNet:
                                dict_to_return_ElasticNet[protein][alpha] = {}
                                for l1 in [0.1, 0.5, .9]:
                                    start_time = time.time()
                                    dict_to_return_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1)
                                    print('Time for ElasticNet in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, network_name, protein, dict_to_return_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_test'], dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                            if do_SVM:
                                start_time = time.time() 
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                                print('Time for default SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                print('Time for linear SVM in NW: ', (time.time() - start_time))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                                print('Time for poly SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                # start_time = time.time()
                                # dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='sigmoid')
                                # print('Time for sigmoid SVM in NW: ', (time.time() - start_time))
                                # df_all_results = df_all_results.append(pd.DataFrame([['SVMsigmoid', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                # pickle.dump(df_all_results, open('output/ML/df_all_results_' + network_name + '.pkl','wb'))
                    if do_Forest: 
                        if n_split>=2:
                            a = [[1,45]]#a = [[10,50,100,200], [1,0.5,0.1], [None, 3,10,20]], [1,3,10]] # parameters for n_estimators=100, max_features=1.0, max_depth=None, min_samples_leaf=1
                            # a = [[120], [0.5], [3], [3]] # best parameters for both datasets
                            b = list(itertools.product(*a))  # all combinations of the parameters
                            for index_param, combi_params in enumerate(b): 
                                # n_estimators = combi_params[0]
                                n_estimators = 100
                                # max_features = combi_params[1]
                                max_features = 'auto'
                                max_depth = 3
                                # min_samples_leaf = combi_params[3]
                                min_samples_leaf = combi_params[0]
                                # param = {'n_estimators':combi_params[0], "max_features":combi_params[1], "max_depth":combi_params[2]}#, "min_samples_leaf":combi_params[3]}
                                class_weight='balanced'
                                start_time = time.time()
                                dict_RandomForest = ML_LinearRegression.Forest_regression(X.transpose(),y, n_split=n_split, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                                print('Time for RandomForest in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Forest', min_samples_leaf, max_depth, use_self_imputed, network_name, protein, dict_RandomForest['r2_test'], dict_RandomForest['r2_train'],  (time.time() - start_time), len(features_to_use), dict_RandomForest['RMSE_test'], dict_RandomForest['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                        
        print('\n##############################################################')
        print('done with the network part')
        print('##############################################################\n')
        pickle.dump(df_all_results, open(path_df_save,'wb'))
        
    # different abundant "bins" of proteins are used as features        
    if for_tenth_abundance:
        # we have ten bins. 
        # in bin 1 the ten percent highest abundant proteins are
        # in bin 2 the ten percent second highest abundant proteins are
        # .... 
        
        # for each tenth highest abundance we use the most
        which_tenth_abundance = 2  # second ten percent highest abundant proteins
        network_name = str(which_tenth_abundance) + 'tenth_abundance'
        path_df_save = 'output/ML/df_all_results_' + network_name + '.pkl'
        df_abundance = pd.DataFrame(index=measured_proteins, columns=['MedianAbundance'])
        for protein in measured_proteins:
            df_abundance['MedianAbundance'][protein] = proteome_no_QC_log[protein].median()
        df_abundance_ordered = df_abundance.sort_values(by='MedianAbundance', ascending=False)
        start_from = (which_tenth_abundance-1)*int(len(list(df_abundance_ordered.index))/10)
        up_to  = (which_tenth_abundance)*int(len(list(df_abundance_ordered.index))/10)
        ten_percent_lowest_abundant = list(df_abundance_ordered.index)[start_from:up_to]
        test = df_abundance_ordered.loc[ten_percent_lowest_abundant]
        
        dict_proteins = {}  # for each protein the best model and parameters are stored. 
        dict_to_return_Lasso = {}
        dict_to_return_SimpleRegression = {}
        dict_to_return_ElasticNet = {}
        dict_to_return_Ridge = {}
        dict_to_return_SVM = {}
        
        for index_protein, protein in enumerate(measured_proteins[from_number_proteins:how_many_proteins]):
            print('\n##############################################################')
            print('Proteins so far: ', counter)
            print('##############################################################\n')
            counter+=1
            dict_proteins[protein] = {}
            dict_to_return_Lasso[protein] = {}
            dict_to_return_SimpleRegression[protein] = {}
            dict_to_return_ElasticNet[protein] = {}
            dict_to_return_Ridge[protein] = {}
            dict_to_return_SVM[protein] = {}
            features_to_use = ten_percent_lowest_abundant.copy()
            if protein in features_to_use:
                features_to_use.remove(protein)
            if len(features_to_use):
                X = proteome_no_QC_log[features_to_use].transpose()
                X = X.apply(pd.to_numeric)
                y = proteome_no_QC_log[protein]
                y = y.apply(pd.to_numeric)
                
                # remove samples, where the protein is not measured:
                not_measured_samples = list(proteome_no_QC_notImputed[protein].index[proteome_no_QC_notImputed[protein].isnull()])
                X.drop(not_measured_samples, axis=1, inplace=True)
                y.drop(not_measured_samples, inplace=True)
                
                if len(X.columns):
                    # y = np.asarray(y).reshape(-1,1)
                    # X = np.asarray(X).reshape(-1,1)
                    number_best_features = int(len(X.columns)/5)
                    if not number_best_features:
                        number_best_features =1
                    r2test = float('nan')
                    model ='model'
                    alpha_to_use = float('nan')
                    l1_to_use = float('nan')
                    n_split = min(5, len(features_to_use))
                    if n_split>=2:
                        for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                            if do_Lasso:
                                start_time = time.time()
                                dict_to_return_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha)
                                print('Time for Lasso in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Lasso[protein][alpha]['r2_test'], dict_to_return_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Lasso[protein][alpha]['RMSE_test'], dict_to_return_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            # dict_to_return_SimpleRegression[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SimpleRegression', alpha=alpha)
                            if do_Ridge:
                                start_time = time.time()
                                dict_to_return_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha)
                                print('Time for Ridge in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_Ridge[protein][alpha]['r2_test'], dict_to_return_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_Ridge[protein][alpha]['RMSE_test'], dict_to_return_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                            
                            if do_ElasticNet:
                                dict_to_return_ElasticNet[protein][alpha] = {}
                                for l1 in [0.1, 0.5, .9]:
                                    start_time = time.time()
                                    dict_to_return_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1)
                                    print('Time for ElasticNet in NW: ', (time.time() - start_time))
                                    df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, network_name, protein, dict_to_return_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_test'], dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_train']]], columns=columns))
                                    pickle.dump(df_all_results, open(path_df_save,'wb'))
                            if do_SVM:
                                start_time = time.time() 
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                                print('Time for default SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                print('Time for linear SVM in NW: ', (time.time() - start_time))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                start_time = time.time()
                                dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                                print('Time for poly SVM in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                                
                                # start_time = time.time()
                                # dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='sigmoid')
                                # print('Time for sigmoid SVM in NW: ', (time.time() - start_time))
                                # df_all_results = df_all_results.append(pd.DataFrame([['SVMsigmoid', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), len(features_to_use), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                                # pickle.dump(df_all_results, open('output/ML/df_all_results_' + network_name + '.pkl','wb'))
                    if do_Forest: 
                        if n_split>=2:
                            a = [[1,45]]#a = [[10,50,100,200], [1,0.5,0.1], [None, 3,10,20]], [1,3,10]] # parameters for n_estimators=100, max_features=1.0, max_depth=None, min_samples_leaf=1
                            # a = [[120], [0.5], [3], [3]] # best parameters for both datasets
                            b = list(itertools.product(*a))  # all combinations of the parameters
                            for index_param, combi_params in enumerate(b): 
                                # n_estimators = combi_params[0]
                                n_estimators = 100
                                # max_features = combi_params[1]
                                max_features = 'auto'
                                max_depth = 3
                                # min_samples_leaf = combi_params[3]
                                min_samples_leaf = combi_params[0]
                                # param = {'n_estimators':combi_params[0], "max_features":combi_params[1], "max_depth":combi_params[2]}#, "min_samples_leaf":combi_params[3]}
                                class_weight='balanced'
                                start_time = time.time()
                                dict_RandomForest = ML_LinearRegression.Forest_regression(X.transpose(),y, n_split=n_split, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                                print('Time for RandomForest in NW: ', (time.time() - start_time))
                                df_all_results = df_all_results.append(pd.DataFrame([['Forest', min_samples_leaf, max_depth, use_self_imputed, network_name, protein, dict_RandomForest['r2_test'], dict_RandomForest['r2_train'],  (time.time() - start_time), len(features_to_use), dict_RandomForest['RMSE_test'], dict_RandomForest['RMSE_train']]], columns=columns))
                                pickle.dump(df_all_results, open(path_df_save,'wb'))
                        
        print('\n##############################################################')
        print('done with the network part')
        print('##############################################################\n')
        pickle.dump(df_all_results, open(path_df_save,'wb'))




    counter = from_number_proteins
    if forAllProteins:
       dict_proteins = {}  # for each protein the best model and parameters are stored. 
       dict_to_return_Lasso = {}
       dict_to_return_SimpleRegression = {}
       dict_to_return_ElasticNet = {}
       dict_to_return_Ridge = {}
       dict_to_return_SVM = {}
       n_split =10
       for protein in measured_proteins[from_number_proteins:how_many_proteins]:
            print('\n##############################################################')
            print('Proteins so far: ', counter)
            print('##############################################################\n')
            counter+=1
            dict_proteins[protein] = {}
            
            X = proteome_no_QC_log.drop(protein, axis=1).transpose()
            X = X.apply(pd.to_numeric)
            y = proteome_no_QC_log[protein]
            y = y.apply(pd.to_numeric)
    
            r2test = float('nan')
            model ='model'
            alpha_to_use = float('nan')
            dict_to_return_Lasso[protein] = {}
            dict_to_return_SimpleRegression[protein] = {}
            dict_to_return_ElasticNet[protein] = {}
            dict_to_return_Ridge[protein] = {}
            dict_to_return_SVM[protein] = {}

            for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                if do_Lasso:
                    start_time = time.time()
                    dict_to_return_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha)
                    print('Time for Lasso no NW: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_Lasso[protein][alpha]['r2_test'], dict_to_return_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time),float('nan'), dict_to_return_Lasso[protein][alpha]['RMSE_test'], dict_to_return_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                    # dict_to_return_SimpleRegression[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SimpleRegression', alpha=alpha)
                    pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                
                if do_Ridge:
                    start_time = time.time()
                    dict_to_return_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha)
                    print('Time for Ridge no NW: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_Ridge[protein][alpha]['r2_test'], dict_to_return_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_Ridge[protein][alpha]['RMSE_test'], dict_to_return_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                
                if do_ElasticNet:
                    dict_to_return_ElasticNet[protein][alpha] = {}
                    for l1 in [0.1, 0.5, .9]:
                        start_time = time.time()
                        dict_to_return_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1)
                        print('Time for ElasticNet no NW: ', (time.time() - start_time))
                        df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, 'NoNW', protein, dict_to_return_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_test'], dict_to_return_ElasticNet[protein][alpha][l1]['RMSE_train']]], columns=columns))
                        pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                if do_SVM:
                    start_time = time.time()
                    dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                    print('Time for default SVM no NW: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                    
                    # dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                    # df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    start_time = time.time()
                    dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                    print('Time for poly SVM no NW: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                    
            # dict_to_return_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='sigmoid')
            # df_all_results = df_all_results.append(pd.DataFrame([['SVMsigmoid', alpha, float('nan'), use_self_imputed, 'NoNW', protein, dict_to_return_SVM[protein][alpha]['r2_test'], dict_to_return_SVM[protein][alpha]['r2_train'],  (time.time() - start_time),float('nan'), dict_to_return_SVM[protein][alpha]['RMSE_test'], dict_to_return_SVM[protein][alpha]['RMSE_train']]], columns=columns))
            
            
            a = [[1,45]]#a = [[100,120,200], [1,0.5,0.1], [None, 3,10,20]]#, [1,3,10]] # parameters for n_estimators=100, max_features=1.0, max_depth=None, min_samples_leaf=1
            # a = [[120], [0.5], [3], [3]] # best parameters for both datasets
            b = list(itertools.product(*a))  # all combinations of the parameters
            for index_param, combi_params in enumerate(b): 
                # n_estimators = combi_params[0]
                n_estimators = 100
                max_features = 'auto'
                max_depth = 3
                # max_features = combi_params[1]
                # max_depth = combi_params[2]
                # min_samples_leaf = combi_params[3]
                min_samples_leaf = combi_params[0]
                class_weight='balanced'
                start_time = time.time()
                # param = {'n_estimators':combi_params[0], "max_features":combi_params[1], "max_depth":combi_params[2]}#, "min_samples_leaf":combi_params[3]}
                dict_RandomForest = ML_LinearRegression.Forest_regression(X.transpose(),y, n_split=n_split, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                print('Time for RandomForest no NW: ', (time.time() - start_time))
                df_all_results = df_all_results.append(pd.DataFrame([['Forest', min_samples_leaf, max_depth, use_self_imputed, network_name, protein, dict_RandomForest['r2_test'], dict_RandomForest['r2_train'],  (time.time() - start_time), float("nan"), dict_RandomForest['RMSE_test'], dict_RandomForest['RMSE_train']]], columns=columns))
                pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
                
    
            

       print('\n##############################################################')
       print('done with the All Proteins part')
       print('##############################################################\n')
       pickle.dump(df_all_results, open('output/ML/df_all_results_noNW.pkl','wb'))
       
    if do_PCA:
       dict_proteins_PCA = {}  # for each protein the best model and parameters are stored. 
       dict_to_return_PCA_Lasso = {}
       dict_to_return_PCA_SimpleRegression = {}
       dict_to_return_PCA_ElasticNet = {}
       dict_to_return_PCA_Ridge = {}
       dict_to_return_PCA_SVM = {}
       n_split =10
       for protein in overlap_measured[from_number_proteins:how_many_proteins]:
            print('\n##############################################################')
            print('Proteins so far: ', counter)
            print('##############################################################\n')
            counter+=1
            dict_proteins_PCA[protein] = {}
            X = proteome_no_QC_log.drop(protein, axis=1).transpose()
            X = X.apply(pd.to_numeric)
            y = proteome_no_QC_log[protein]
            y = y.apply(pd.to_numeric)
    
            r2test = float('nan')
            model ='model'
            alpha_to_use = float('nan')
            dict_to_return_PCA_Lasso[protein] = {}
            dict_to_return_PCA_SimpleRegression[protein] = {}
            dict_to_return_PCA_ElasticNet[protein] = {}
            dict_to_return_PCA_Ridge[protein] = {}
            dict_to_return_PCA_SVM[protein] = {}
            
            
            for alpha in alphas:#np.logspace(-4, -0.5, 10): #np.arange(0,1.1,0.1):
                if do_Lasso:
                    start_time = time.time()
                    dict_to_return_PCA_Lasso[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Lasso', alpha=alpha, do_PCA=do_PCA)
                    print('Time for Lasso PCA: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['Lasso', alpha, float('nan'), use_self_imputed, 'PCA', protein, dict_to_return_PCA_Lasso[protein][alpha]['r2_test'], dict_to_return_PCA_Lasso[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_Lasso[protein][alpha]['RMSE_test'], dict_to_return_PCA_Lasso[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
                
                # dict_to_return_PCA_SimpleRegression[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SimpleRegression', alpha=alpha)
                if do_Ridge:
                    start_time = time.time()
                    dict_to_return_PCA_Ridge[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='Ridge', alpha=alpha, do_PCA=do_PCA)
                    print('Time for Ridge PCA: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['Ridge', alpha, float('nan'), use_self_imputed, 'PCA', protein, dict_to_return_PCA_Ridge[protein][alpha]['r2_test'], dict_to_return_PCA_Ridge[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_Ridge[protein][alpha]['RMSE_test'], dict_to_return_PCA_Ridge[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
              
                if do_ElasticNet:
                    dict_to_return_PCA_ElasticNet[protein][alpha] = {}
                    for l1 in [0.1, 0.5, .9]:
                        start_time = time.time()
                        dict_to_return_PCA_ElasticNet[protein][alpha][l1] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='ElasticNet', alpha=alpha, l1_ratio=l1, do_PCA=do_PCA)
                        print('Time for ElasticNet PCA: ', (time.time() - start_time))
                        df_all_results = df_all_results.append(pd.DataFrame([['ElasticNet', alpha, l1, use_self_imputed, 'PCA', protein, dict_to_return_PCA_ElasticNet[protein][alpha][l1]['r2_test'], dict_to_return_PCA_ElasticNet[protein][alpha][l1]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_ElasticNet[protein][alpha]['RMSE_test'], dict_to_return_PCA_ElasticNet[protein][alpha]['RMSE_train']]], columns=columns))
                        pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
                
                if do_SVM:
                    start_time = time.time()
                    dict_to_return_PCA_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha)
                    print('Time for default SVM PCA: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['SVM', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_PCA_SVM[protein][alpha]['r2_test'], dict_to_return_PCA_SVM[protein][alpha]['r2_train'],  (time.time() - start_time),float('nan'), dict_to_return_PCA_SVM[protein][alpha]['RMSE_test'], dict_to_return_PCA_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
                    
                    # dict_to_return_PCA_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='linear')
                    # df_all_results = df_all_results.append(pd.DataFrame([['SVMlinear', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_PCA_SVM[protein][alpha]['r2_test'], dict_to_return_PCA_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_SVM[protein][alpha]['RMSE_test'], dict_to_return_PCA_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    
                    start_time = time.time()
                    dict_to_return_PCA_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='poly')
                    print('Time for poly SVM PCA: ', (time.time() - start_time))
                    df_all_results = df_all_results.append(pd.DataFrame([['SVMpoly', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_PCA_SVM[protein][alpha]['r2_test'], dict_to_return_PCA_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_SVM[protein][alpha]['RMSE_test'], dict_to_return_PCA_SVM[protein][alpha]['RMSE_train']]], columns=columns))
                    pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
                    
            # dict_to_return_PCA_SVM[protein][alpha] = ML_LinearRegression.RMSE_regression(X,y, n_split=n_split, what_sort='SVM', alpha=alpha, kernel='sigmoid')
            # df_all_results = df_all_results.append(pd.DataFrame([['SVMsigmoid', alpha, float('nan'), use_self_imputed, network_name, protein, dict_to_return_PCA_SVM[protein][alpha]['r2_test'], dict_to_return_PCA_SVM[protein][alpha]['r2_train'],  (time.time() - start_time), float('nan'), dict_to_return_PCA_SVM[protein][alpha]['RMSE_test'], dict_to_return_PCA_SVM[protein][alpha]['RMSE_train']]], columns=columns))
     
       print('\n##############################################################')
       print('done with the PCA part')
       print('##############################################################\n')
       pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
       
pickle.dump(df_all_results, open('output/ML/df_all_results.pkl','wb'))
