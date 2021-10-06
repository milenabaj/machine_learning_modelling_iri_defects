"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from utils.data_transforms import *
from utils.plotting import *
from utils.data_loaders import *
from utils.analysis import *
import argparse
from utils.analysis import *
import tsfel
import pickle
from utils.analysis import *
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter
from math import ceil, floor
import gc, os, sys, glob             
import scipy.stats
import tsfel
from scipy import stats, interpolate
import sklearn
 
if __name__=='__main__':
    
    # ======================= #
    # Setup
    # ======================= #  

    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    
    parser.add_argument('--GM_task_id', default =  5683,
                        help='Unique taskID defining the GM trip.')
    
    
    explore_ts = False
    explore_fe = True
    
    if explore_ts:
        f = 'results-5683-passes-window-100-step-10/matching/aran-V/aligned_GM-5683_DRD-974a5c25-ee35-43c6-a2d5-2a486ec6ab0e_p79_GM_aran.pickle'
    
        df = pd.read_pickle(f)
        print(df.shape)
        
        target = 'KPI'
        target_col = [col for col in df.columns if (not col.startswith('GM') and not 'lat_' in col and not 'lon_' in col and not 'TS_or_Distance' in col and not 'street' in col)]
        pothole_col = ['PotholeAreaAffectedLow','PotholeAreaAffectedMed', 'PotholeAreaAffectedHigh','PotholeAreaAffectedDelam'] # only 'PotholeAreaAffectedHigh' has a variation
        crack_trans_col = ['CracksTransverseSmall','CracksTransverseMed','CracksTransverseLarge','CracksTransverseSealed']
        crack_long_col = ["CracksLongitudinalSmall","CracksLongitudinalMed","CracksLongitudinalLarge","CracksLongitudinalSealed"]
        crack_alig_col = ["AlligCracksSmall","AlligCracksMed","AlligCracksLarge"]
        all_t_col = pothole_col + crack_trans_col + crack_long_col + crack_alig_col  
        gm_col = [col for col in df.columns if col.startswith('GM')]
       
        plt.title('KPI')
        df['KPI'].hist()
        
        plt.title('DI')
        df['DI'].hist()
        
        bad_df = df[df['KPI']>5]
        bad_df.reset_index(inplace=True)
        good_df = df[df['KPI']<1.5]
        good_df.reset_index(inplace=True)
        
        # 'GM.obd.odo.value'
        var = 'GM.acc.xyz.z'
        m = np.inf
        target_name = 'DI'
        plot_type(df, var, lower_limit  = 2.4, is_bad=True, n_plots=m, 
                  target_name = target_name)    
        plot_type(df, var, upper_limit = 0.5, n_plots=m, 
                  target_name = target_name)  
            

        
    if explore_fe:
        f = 'results-5683-passes-window-100-step-10/feature-extraction-resampled/H/feature-extraction-prepared-files-resampled/DRDGM_trip-5683_extracted-features_resampled_full_pass-0.pickle'
        
        df = pd.read_pickle(f)
        print(df.shape)
        
        target_col = [col for col in df.columns if (not col.startswith('GM') and not 'lat_' in col and not 'lon_' in col and not 'TS_or_Distance' in col and not 'street' in col)]
        pothole_col = ['PotholeAreaAffectedLow','PotholeAreaAffectedMed', 'PotholeAreaAffectedHigh','PotholeAreaAffectedDelam'] # only 'PotholeAreaAffectedHigh' has a variation
        crack_trans_col = ['CracksTransverseSmall','CracksTransverseMed','CracksTransverseLarge','CracksTransverseSealed']
        crack_long_col = ["CracksLongitudinalSmall","CracksLongitudinalMed","CracksLongitudinalLarge","CracksLongitudinalSealed"]
        crack_alig_col = ["AlligCracksSmall","AlligCracksMed","AlligCracksLarge"]
        all_t_col = pothole_col + crack_trans_col + crack_long_col + crack_alig_col  
        gm_col = [col for col in df.columns if col.startswith('GM')]
        rut = ['p79.RutDepthLeft','p79.RutDepthRight','rutting']
        iris = ['p79.IRI5','p79.IRI21','iri']
        x_lab =  'GM.acc.xyz.z-0_Maxmin diff'
        df['rutting'] = (df['p79.RutDepthLeft'] + df['p79.RutDepthRight'])/2
        df['iri'] = (df['p79.IRI5'] + df['p79.IRI21'])/2
   
        #sns.pairplot(df.head(1000), vars = pothole_col  + target_col)
        
        std =  'GM.acc.xyz.z-0_Maxmin diff'
        for col in crack_trans_col:
            col = 'iri'
            plt.figure()
            
            x = df[std]
            y = df[col]
                        
            plt.scatter(x,y)
            
            col = 'IRI'
            plt.xlabel(std)
            plt.ylabel(col)
            plt.title(col)
            print('\n=====',col)
            print(var.describe())
            
            
        for col in rut:
            plt.figure()
            y = df[col]
            x = df[x_lab]
            plt.scatter(x,y)
            plt.xlabel(x_lab)
            #plt.xlim([25, 50])
            plt.ylabel(col)
            plt.title(col)
            print('\n=====',col)
            print(var.describe())
            
        for col in all_t_col:
            plt.figure(figsize=(5,5))
            var = df[col]
            t = df[std]
            plt.scatter(var,t)
            plt.xlabel(col)
            #plt.xlim([25, 50])
            plt.ylabel(std)
            plt.title(col)
            print('\n=====',col)
            print(var.describe()) 
                    
    #savefig("output.png")
    sys.exit(0)
    '''
    # Parse arguments
    args = parser.parse_args()
    GM_trip_id  = args.GM_task_id 
    res_dir = 'results-{0}'.format(GM_trip_id) 
    target_name = 'IRI_mean_end'
    do_latex_table = False
     
    # Get matching DRD trip id for this GM trip id
    #DRD_trip_id = get_matching_DRD_trip_id(GM_trip_id)  
    bins = [0,0.6,1.2,2.4,4.8,9.6]
    
    #feats = ['acc.xyz.z','obd.spd_veh.value']
    feats =  ['acc.xyz.z']
    
    if do_latex_table:
        f = 'results-5683-passes-window-100-step-10/feature-selection-resampled/regression_fmax-68/GMDRD_5683_selectedfeatures_inputvars-_fmax-68_nt-100_x_train_regression_feats_info.pickle'
        feats = pd.read_pickle(f)
        feats = feats[['Added Feature', 'MSE (subset)']]
        feats.to_latex('reg_fs_table.tex', columns = feats.columns, index = True, 
                                float_format = lambda
                                x: '%.2e' % x, label = 'table:reg_fs',  
                                header=[ format_col(col) for col in feats.columns] ,escape=False)
        
    '''
    '''    
    # =====================================================================   #
    # Plot all sequences in 1 figure
    # =====================================================================   #
    for var in feats:
        plt.figure()
        plt.title(var)
        
        # Loop over df's
        for i in l:
            t = i[0] 
            df = i[1]
            if t=='good':
                c='black'
            elif t=='med':
                c='green'
            elif t=='bad':
                c='red'
               
            df_len = df.shape[0]
            if df_len>=50:
                n_rows = 50
            else:
                n_rows=df_len
            n_rows=5
            print(t, c, n_rows, df_len)
            
            # Loop over rows
            for row in range(0,n_rows):
                iri = df['IRI'].iloc[row]
                seq = df[var].iloc[row]
                
                if row==0:
                    plt.plot(seq, label=t, linestyle = 'None', marker='.',color = c, alpha=0.5)
                else:
                    plt.plot(seq, linestyle = 'None', marker='.',color = c, alpha=0.5)
    
                plt.ylabel(var)
    
                #compute_features_per_series(seq, cfg)
                
            plt.tight_layout()
            plt.legend()
            #break
            #df[colname_tsfel] = df[var].apply(lambda seq: compute_features_per_series(seq, cfg) )
            #keep_cols.append(colname_tsfel)
    


    # =====================================================================   #
    # Plot sequences, multiple figures: 1 bad vs 1 med vs 1 good
    # =====================================================================   #    
        
    n_rows= bad_df.shape[0]
    n_rows=10

    # Loop over rows
    for row in range(0,n_rows):
        plt.figure()
        
        # dfs
        for i in l:
            t = i[0] 
            df = i[1]
            if t=='good':
                c='black'
            elif t=='med':
                c='green'
            elif t=='bad':
                c='red'
                
            iri = df['IRI'].iloc[row]
            seq = df[var].iloc[row]
            mean = seq.mean()
            med =  np.median(seq)
            std = np.std(seq)
            mad = stats.median_absolute_deviation(seq)
            abs_min_max = seq.max()-seq.min()
            pp = tsfel.features.positive_turning(seq)
            
            # Plot params
            plt.plot(seq, marker='.',color = c, alpha=0.5, label=t+ ': pp' + str(pp))
            #plt.axhline(y=abs_min_max, linestyle='--',color = c, label=t+ ': diff')
            #plt.axhline(y=std, linestyle='dotted',color = c, label=t+ ': std')
            #plt.axhline(y=mad, linestyle='--',color = c, label=t+ ': mad')
            #plt.axhline(y=mean, linestyle='dotted',color = c, label=t+ ': mean')
            #plt.axhline(y=med, linestyle='--',color = c, label=t+ ': median')
            plt.ylabel(var)

            
        plt.tight_layout()
        #plt.axhline(y=1, linestyle='--',color = '#fb6500', label='1')
        plt.legend()

            #break
    sys.exit(0)
    '''
    # =====================================================================   #
    # Make plots
    # =====================================================================   #          
    for var in ['acc.xyz.z']:
        plt.figure()
        
        for row in range(0,bad_df.shape[0]):
            bad = bad_df.iloc[row]
            bad_data, bad_p = empirical_cdf(bad[var])
            if row==0:
                plt.plot(bad_data,bad_p, label='IRI > {0:.1f}'.format(iri_bad), color = 'red')
            else:
                plt.plot(bad_data,bad_p, color = 'red') 
        
        
        for row in range(0,good_df.shape[0]):  
            good = good_df.iloc[row]
            good_data, good_p = empirical_cdf(good[var])
            if row==0:
                plt.plot(good_data,good_p, label='IRI < {0:.1f}'.format(iri_good),color = 'blue',alpha=0.5)
            else:
                plt.plot(good_data,good_p, color = 'blue', alpha=0.5)  
        
                
        for row in range(0,med_df.shape[0]):  
            med = med_df.iloc[row]
            med_data, med_p = empirical_cdf(med[var])
            if row==0:
                plt.plot(med_data,med_p, label='IRI > {0:.1f} & IRI < {1:.1f}'.format(iri_good, iri_med),color = 'green',alpha=0.3)
            else:
                plt.plot(med_data,med_p,color = 'green',alpha=0.3)
                

        plt.xlabel(var)
        plt.ylabel('percentiles')

        plt.axhline(y=0.75, linestyle='--',color = '#fb6500')
        plt.axhline(y=0.5, linestyle='--',color = '#fb6500')
        plt.axhline(y=0.25, linestyle='--',color = '#fb6500')
        plt.tight_layout()
        plt.legend()
       
    # Min
    bad = bad_df[var].apply(lambda row: row.min())
    med = med_df[var].apply(lambda row: row.min())   
    good = good_df[var].apply(lambda row: row.min())
    
    for var in ['acc.xyz.z']:
       plt.figure()
       
       bad_data, bad_p = empirical_cdf(bad)
       plt.plot(bad_data,bad_p, label='IRI > {0:.1f}'.format(iri_bad), color = 'red')

       med_data, med_p = empirical_cdf(med)
       plt.plot(med_data,med_p, label='IRI > {0:.1f}'.format(iri_med), color = 'green')

       good_data, good_p = empirical_cdf(good)
       plt.plot(good_data,good_p, label='IRI > {0:.1f}'.format(iri_good), color = 'blue')

       plt.xlabel(var+' min')
       plt.ylabel('percentiles')

       plt.axhline(y=0.75, linestyle='--',color = '#fb6500')
       plt.axhline(y=0.5, linestyle='--',color = '#fb6500')
       plt.axhline(y=0.25, linestyle='--',color = '#fb6500')
       plt.tight_layout()
       plt.legend()
        
     # Percentile
    bad = bad_df[var].apply(lambda row: np.percentile(row, 5))
    med = med_df[var].apply(lambda row: np.percentile(row, 5)) 
    good = good_df[var].apply(lambda row: np.percentile(row, 5))
     
    for var in ['acc.xyz.z']:
        plt.figure()
        
        bad_data, bad_p = empirical_cdf(bad)
        plt.plot(bad_data,bad_p, label='IRI > {0:.1f}'.format(iri_bad), color = 'red')
    
        med_data, med_p = empirical_cdf(med)
        plt.plot(med_data,med_p, label='IRI > {0:.1f}'.format(iri_med), color = 'green')
    
        good_data, good_p = empirical_cdf(good)
        plt.plot(good_data,good_p, label='IRI > {0:.1f}'.format(iri_good), color = 'blue')
    
        plt.xlabel(var+' percentile')
        plt.ylabel('percentiles')
    
        plt.axhline(y=0.75, linestyle='--',color = '#fb6500')
        plt.axhline(y=0.5, linestyle='--',color = '#fb6500')
        plt.axhline(y=0.25, linestyle='--',color = '#fb6500')
        plt.tight_layout()
        plt.legend()
       
    # Ent
    bad['ent'] = bad_df[var].apply(lambda row: ent(pd.Series(row)))
    med['ent'] = med_df[var].apply(lambda row: ent(pd.Series(row)))
    good['ent'] = good_df[var].apply(lambda row: ent(pd.Series(row)))
     
    print(df.memory_usage())


