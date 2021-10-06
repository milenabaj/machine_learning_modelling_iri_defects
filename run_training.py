"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from utils.data_transforms import *
from utils.plotting import *
import argparse, json
from utils.analysis import *
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer 
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import classification_report, plot_confusion_matrix
from utils.analysis import *
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter
from math import ceil, floor
import gc, os, sys, glob             
from sklearn.model_selection import  TimeSeriesSplit, GridSearchCV
import seaborn as snsls 
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import linear_model
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN

if __name__=='__main__':
    
    # ======================= #
    # User arguments
    # ======================= #  

    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    
    # Select what to use 
    parser.add_argument('--routes', action = 'append', help='Process all routes.')
    parser.add_argument('--trip', type=int, help='Process this trip only. By default, used is pass 0 only.') 
    parser.add_argument('--feats', action = 'append', help='Consider those features. Use: all to use acc in 3-axis and speed or pass a combination of acc_x, acc_y, acc_z and speed.')    
    parser.add_argument('--recreate', action="store_true", help = 'Recreate files, even if present. If False and the files are present, the data will be loaded from them.')
    parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
   
    # Model options
    parser.add_argument('--do_reg', default=False, action='store_true')
    parser.add_argument('--do_class', default=False, action='store_true')
    parser.add_argument('--rm', action='append', type=str, 
                        help='Pass a list of regression models. If not passed, all available will be run.')
    parser.add_argument('--cm', action='append', type=str, 
                        help='Pass a list of classification models. If not passed, all available will be run.')
    
    # Data type
    parser.add_argument('--is_p79', action='store_true', help = 'If this is p79 data, pass true.')
    parser.add_argument('--is_aran', action='store_true', help = 'If this is aran data, pass true.')
    parser.add_argument('--aran_target',  default='DI')
    parser.add_argument('--predict_mode', action="store_true", help = 'Run in prediction mode. P79 is not needed, GM data will be split into windows.') 
    
    # Load additional sensors
    parser.add_argument('--load_add_sensors', action='store_true', help = 'Load additional sensors.')  
    
    # Random seed
    parser.add_argument('--rs', default =  47,
                        help='Random state')
    
    # ======================= #
    # Set options
    # ======================= #
    args = parser.parse_args()
     
    routes = args.routes
    trip = args.trip 
    recreate = args.recreate
    json_file = args.json
    
    is_aran = args.is_aran
    is_p79 = args.is_p79
    aran_target = args.aran_target
    random_state = int(args.rs)
    predict_mode = args.predict_mode
    
    load_add_sensors = args.load_add_sensors
    
    # Default
    window_size = 100
    step = 10
    do_explore = True
    do_plots = True
    do_fs_plots = True
    pca_perc = 0.99
    do_reg = True #do_reg = args.do_reg
    
    # Input features
    use_feats = []
    
    # Default
    if not args.feats:
        use_feats = ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z'] # use those for final files
        out_name = 'speed-acc_x-acc_y-acc-z'
    # Else take from arguments
    else:
        out_name = ''
        for feat in args.feats:
            if feat=='all':
                use_feats =  ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z'] 
                out_name = 'speed-acc_x-acc_y-acc-z'
                break
            if feat=='acc_x':
                use_feats.append('GM.acc.xyz.x')
                out_name = out_name+'-'+feat
            elif feat=='acc_y':
                use_feats.append('GM.acc.xyz.y')
                out_name = out_name+'-'+feat
            elif feat=='acc_z':
                use_feats.append('GM.acc.xyz.z')
                out_name = out_name+'-'+feat
            elif feat=='speed':
                use_feats.append('GM.obd.spd_veh.value')
                out_name = out_name+'-'+feat
    if out_name.startswith('-'):
        out_name = out_name[1:]
    
     
    # Additional sensors to load
    if load_add_sensors:
        out_name = out_name+'-steering-wheelpressure-yaw-traccons'

    # Try to find a route if a trip is given, but route not given
    if args.trip and not routes:
        
        # Load json file with route info
        with open(json_file, "r") as f:
            route_data = json.load(f)
        
        # Try to find it from json file and the given trip
        for route_cand in route_data.keys():
            if trip in route_data[route_cand]['GM_trips']:
                routes = [route_cand]
                break
            
        # If still not found, exit
        if not routes:        
            print('Please pass the route so the DRD trip can be found.')
            sys.exit(0)    
       
        
    # If a specific trip nor a route is given, then load the default routes
    elif not args.trip:
        routes = ['CPH1_HH', 'CPH1_VH']
    combined_route_name = '_'.join(routes)
   
    # Regression
    if args.rm:
        reg_models = args.rm
        outfile_suffix = 'all'
    else:
        reg_models = ['linear','lasso','ridge','elastic_net','kNN','random_forest','SVR_rbf','ANN']
        #reg_models = ['dummy', 'lasso','random_forest']
        
    outfile_suffix = '_'.join(reg_models)
    reg_scoring = 'r2' #, 'r' 'neg_mean_squared_error'
          
    #0-0.6 (excellent, difficult to obtained on newly paved roads)
    #0.6 -0.9 ( very good, normally what we measure on resurfaced roads)
    #0.9 - 1.2( good, 0.9 is the average in DRD roads)
    #1.2 - 1.6 ( poor)
    #1.6 - 2.4 (bad)
    #2.4 - 15 very bad (even if high IRI can be seen over bridge joints)

    # Classification
    do_class = args.do_class
    do_class = False
    bins =  [0,0.9,1.6,2.5,20] 
    if args.cm:
        class_models = args.cm
    else:
        class_models = ['dummy','naive_bayes', 'kNN','logistic_regresion','random_forest','SVC_rbf','ANN']
        #class_models = ['dummy']
    outfile_suffix = '_'.join(class_models)
    outfile_suffix  = '{0}_{1}'.format('_'.join(str(b) for b in bins), outfile_suffix)
 
    # ========================== #
    # Input and output directories
    # ========================== #        
    # Input directory
    if is_aran:
        if predict_mode:
            in_dir = ''
            in_fs_dir = in_dir
            in_model_dir = 'data/aligned_GM_p79_data_window-{0}-step-{1}-modelling-{2}/{3}/chunks/regression'.format(window_size, step, out_name, combined_route_name)
        else:
            in_fs_dir = 'data/aligned_GM_ARAN_data_window-{0}-step-{1}-feature-selection-{2}/{3}'.format(window_size, step, out_name, combined_route_name)
        target_name = aran_target
    elif is_p79:
        if predict_mode:
            in_dir = 'data/predict_mode_GM_data_window-{0}-step-{1}-feature-extraction/{3}'.format(window_size, step, out_name, combined_route_name)
            in_fs_dir = in_dir
            in_model_dir = 'data/aligned_GM_p79_data_window-{0}-step-{1}-modelling-{2}/{3}/chunks/regression'.format(window_size, step, out_name, combined_route_name)
        else:
            in_fs_dir = 'data/aligned_GM_p79_data_window-{0}-step-{1}-feature-selection-{2}/{3}'.format(window_size, step, out_name, combined_route_name)
        target_name = 'IRI_mean'
    else:
        print('Set either is_aran or is_iri to True.')
        sys.exit(0)
        
    if load_add_sensors:
        in_fs_dir = in_fs_dir + '_add_sensors'
    
    if aran_target!='DI':
        in_fs_dir = in_fs_dir+'_individual_defects_'+ aran_target
        
    # 1 trip is given
    #if args.trip:
    #    in_fs_dir = in_fs_dir + '/chunks'
        
    # Create output directory
    out_dir = in_fs_dir.replace('feature-selection','modelling')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  
    
    out_dir_reg = '{0}/regression'.format(out_dir)
    if not os.path.exists(out_dir_reg):
        os.makedirs(out_dir_reg)  
    
    out_dir_class = '{0}/classification'.format(out_dir)
    if not os.path.exists(out_dir_class):
        os.makedirs(out_dir_class)  
        
    print('p79 data? ', is_p79)
    print('Aran data? ', is_aran)
    print('Trip: ', trip)
    print('Input directory: ', in_fs_dir)
    print('Output directory: ', out_dir)
    time.sleep(3)
    
    # ========================== #
    # Load data
    # ========================== #
    if predict_mode:
        filename = glob.glob('{0}/route-{1}_trips-{2}.pickle'.format(in_dir, combined_route_name, trip))[0]
        X = pd.read_pickle(filename)
               
        # Scale data (todo: load the train scaler and scale the data)
        scaler = StandardScaler()

    else:
        # Trainvalid df
        trainvalid_filename = glob.glob('{0}/*train_regression.pickle'.format(in_fs_dir))[0]
        trainvalid_df = pd.read_pickle(trainvalid_filename)
        X_trainvalid = trainvalid_df.drop([target_name],axis=1) #REMOVE
        y_trainvalid =  trainvalid_df[target_name]
        
         # Selected features
        f_maxsel = X_trainvalid.shape[1] - 1
        cols = X_trainvalid.columns 
        print(f_maxsel)
        print(cols)
        
        # Test df
        test_filename =  trainvalid_filename.replace('train','test')
        test_df = pd.read_pickle(test_filename)
        X_test = test_df.drop([target_name],axis=1) #REMOVE
        y_test =  test_df[target_name]  
      
        # Scale data
        scaler = StandardScaler()
        X_trainvalid = pd.DataFrame(scaler.fit_transform(X_trainvalid), columns=cols)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=cols)
                
        #PCA
        pca = PCA(pca_perc)
        X_trainvalid_pca = pca.fit_transform(X_trainvalid)
        X_trainvalid_pca = pd.DataFrame(data =  X_trainvalid_pca)
        X_test_pca = pca.transform(X_test)
        X_test_pca = pd.DataFrame(X_test_pca)
    
        plt.title('Train-Valid')
        y_trainvalid.hist()
        plt.savefig('trainvalid.pdf')
        
        plt.title('Test')
        y_test.hist()
        plt.savefig('all.pdf')

    #f = glob.glob('{0}/*_regression.pickle'.format(in_fs_dir))[0]
    #df = pd.read_pickle(f)
    # ========================== #
    # Regression
    # ========================== #
    if do_reg:
             
        # Output directory 
        out_dir = out_dir_reg            
        out_dir_plots = '{0}/plots'.format(out_dir)
        if not os.path.exists(out_dir_plots):
            os.makedirs(out_dir_plots)   
            
            
        if do_fs_plots:
            feats = pd.read_pickle(trainvalid_filename.replace('.pickle','_feats_info.pickle'))
            plot_fs(feats.index, res = feats['MSE (subset)'], out_dir = out_dir_plots, var_label='MSE',filename='feature_selection.pdf')
            
            # Latex table
            feats=feats[0:f_maxsel]
            feats.insert (1, "Description", np.nan*np.ones(f_maxsel))
            #feats = feats[['Added Feature', 'Description','MSE (subset)']]
            feats = feats[['Added Feature', 'Description']]
            feats.rename({'Added Feature':'Feature'}, inplace=True, axis=1)            
            feats['Feature'] =  feats['Feature'].apply(lambda row: row.replace('Acceleration','Acc'))
            feats['Feature'] =  feats['Feature'].apply(lambda row: row.replace('Vehicle Speed','Speed'))
            #plot_feature_importance(feats['MSE (subset)'], feats['Feature'], out_dir = out_dir_plots, filename='fs_mse.pdf',plot_type='mse')
            
            reg_feat_file = out_dir_reg + '/regression_features_table.tex'
            feats.to_latex( reg_feat_file, columns = feats.columns, index = False, 
                            float_format = lambda
                            x: '%.2e' % x, label = 'table:reg_fs',  
                            header=[ format_col(col) for col in feats.columns] ,escape=False)
            
            
            '''
            feats1 = feats[0:13]
            feats1.reset_index(inplace=True, drop = True)
            feats2 = feats[13:]
            feats2.reset_index(inplace=True, drop = True)
            f = pd.concat([feats1, feats2], axis=1)
            f.to_latex('reg_fs_splittable.tex', index = False, 
                            float_format = lambda
                            x: '%.2e' % x, label = 'table:reg_fs',  
                            header=[ format_col(col) for col in f.columns] ,escape=False)'''
            
    
        
        # explore
        if do_explore:
            
            out_dir_explore = '{0}/exploration_plots'.format(out_dir)
            if not os.path.exists(out_dir_explore):
                os.makedirs(out_dir_explore)   
                
            for col in X_trainvalid.columns:
                plt.figure()
                var = X_trainvalid[col]
                plt.scatter(var, y_trainvalid)
                plt.title(col)
                plot_filename = '{0}/{1}_vs_target.png'.format(out_dir_explore, col)
                plt.savefig(plot_filename)
             
            plt.close('all')
            
            # Plot IRI hist
            plt.figure()
            plt.hist(y_trainvalid)
            plt.title('Train')
            plt.xlabel('IRI (m/km)')
            
            plt.figure()
            plt.hist(y_test)
            plt.title('Test')
            plt.xlabel('IRI (m/km)')
            
        plt.close('all')
        
        # Model
        cols = ['Model','R2','MAE','RMSE','MRE']
        train_results = pd.DataFrame(columns = cols)
        train_results_pca = pd.DataFrame(columns = cols)
        test_results = pd.DataFrame(columns = cols)
        test_results_pca = pd.DataFrame(columns = cols)
        
        for row, model in enumerate(reg_models):
            print('Model: ', model)
     
            # Prediction stage
            if predict_mode:
                model_path = '{0}/best_model_{1}.pickle'.format( in_model_dir, model)
                model = pickle.load(open(model_path, 'rb'))
                
                # remove later!!
                n = model.coef_.shape[0]
                X = X.iloc[:, :n]
                
                # Predict
                y_pred = model.predict(X)
                print('Predictions: ',y_pred)
                sys.exit(0) #the end if in predict_mode
                
            # Define the model
            if model=='ANN':
                rf, parameters, model_title = get_regression_model(model,f_maxsel, random_state = random_state, use_default = True, is_pca = False)  
                rf_pca, parameters_pca, model_title_pca = get_regression_model(model, f_maxsel, random_state = random_state, use_default = True, is_pca = True)
            else:
                rf, parameters, model_title = get_regression_model(model,f_maxsel, random_state = random_state, is_pca = False)  
                rf_pca, parameters_pca, model_title_pca = get_regression_model(model, f_maxsel, random_state = random_state, is_pca = True)
                
            # Grid search results
            grid_search_results = {}
            
            # Load GS results
            reg_gs_file = '{0}/grid_search_results_{1}_pca_{2}.pickle'.format(out_dir_reg, model, pca_perc) 
            if not recreate and os.path.exists(reg_gs_file):
                clf = pickle.load(open(reg_gs_file,'rb')) 
                print('Loaded grid search results from: {0}'.format(reg_gs_file))
                
                rf = clf[model].best_estimator_
                rf_pca = clf[model+'_PCA'].best_estimator_
            # Do Grid search
            else:
                # No PCA
                clf, rf = grid_search(rf, parameters, X_trainvalid, y_trainvalid, score=reg_scoring) #
                grid_search_results[model] = clf
                
                # PCA
                clf_pca, rf_pca = grid_search(rf_pca, parameters_pca, X_trainvalid_pca, y_trainvalid, score=reg_scoring)
                grid_search_results[model+'_PCA'] = clf_pca
                
                grid_search_results['params_grid'] = parameters
            
                # Save GS results
                with open(reg_gs_file, 'wb') as handle:
                    pickle.dump(grid_search_results, handle, protocol=4)
                    print('Wrote grid search result to: {0}'.format(reg_gs_file))
             
            # Write the best model
            model_out_file = '{0}/best_model_{1}.pickle'.format(out_dir, model)
            with open(model_out_file, 'wb') as handle:
                    pickle.dump(rf, handle, protocol=4)
                    print('Wrote best model {0} to: {1}'.format(model, model_out_file))
            
            # Write the best PCA model
            pca_model_out_file = '{0}/best_model_pca_{1}.pickle'.format(out_dir, model)
            with open(pca_model_out_file, 'wb') as handle:
                    pickle.dump(rf_pca, handle, protocol=4)
                    print('Wrote best PCA model {0} to: {1}'.format(model,  pca_model_out_file))

            # Best params
            p = rf.get_params()
            p1 = rf_pca.get_params()
                
            # Prediction of the tuned model
            labels = list(range(0, len(bins)-1))
            y_trainvalid_pred,  y_test_pred  = get_regression_predictions(X_trainvalid, y_trainvalid, X_test, y_test, rf, 
                                                                       train_results, test_results, model_title, row, labels)
            
            y_trainvalid_pred_pca,  y_test_pred_pca = get_regression_predictions(X_trainvalid_pca, y_trainvalid, X_test_pca, y_test, rf_pca, 
                                                                                    train_results_pca, test_results_pca, model_title, row, labels)

            if do_plots:
                       
                iri_min = y_trainvalid.min()
                iri_max = y_trainvalid.max() 
                
                # Plot train regression
                plot_regression_true_vs_pred(y_trainvalid, y_trainvalid_pred, title='Train: {0}'.format(model_title),
                                         out_dir = out_dir_plots, filename = '{0}_train'.format(model))
      
                # Plot train (PCA) regression 
                plot_regression_true_vs_pred(y_trainvalid, y_trainvalid_pred_pca, title='Train (PCA): {0}'.format(model_title),
                                         out_dir = out_dir_plots, filename = '{0}_train_pca'.format(model))
               
                # Plot test regression
                plot_regression_true_vs_pred(y_test, y_test_pred, title= 'Test: {0}'.format(model_title),rrr
                                             out_dir = out_dir_plots, filename = '{0}_test'.format(model))
               
                # Plot test (PCA) regression
                plot_regression_true_vs_pred(y_test, y_test_pred_pca, title= 'Test (PCA): {0}'.format(model_title), 
                                             out_dir = out_dir_plots, filename = '{0}_test_pca'.format(model))
                # Feature importance
                if model=='random_forest':
                     feat_names_df = pd.DataFrame(X_trainvalid.columns,columns =['name'])
                     feat_names_df.name = feat_names_df.name.apply(lambda row: get_var_name(row,short = True))
                     feat_names_df.name = feat_names_df.name.apply(lambda row: row.replace('GM.obd.spd_veh.value-0_','Vehicle speed '))
                     rf = clf['random_forest'].best_estimator_
                     plot_feature_importance(rf.feature_importances_, feat_names_df.name, out_dir = out_dir_plots)
                  
                
        models_string = '-'.join(reg_models)
        
        # Get the results table in latex
        train_results = train_results.merge(train_results_pca, on='Model', suffixes=('',' (PCA)'))
        tex_train_table_name = '{0}/regression_table_train_{1}.tex'.format(out_dir_reg, models_string)
        train_results.to_latex(tex_train_table_name, columns = train_results.columns, index = False,
                               float_format = lambda x: '%.2f' % x, label = 'table:iri_reg_pred',  
                              header=[ format_col(col) for col in train_results.columns] ,escape=False)
        print('Latex table (train): ',tex_train_table_name)
        
        test_results = test_results.merge(test_results_pca, on='Model', suffixes=('',' (PCA)'))
        tex_test_table_name = '{0}/regression_table_test_{1}.tex'.format(out_dir_reg, models_string)
        test_results.to_latex(tex_test_table_name, columns = test_results.columns, index = False, 
                              float_format = lambda
                              x: '%.2f' % x, label = 'table:iri_reg_pred',  
                              header=[ format_col(col) for col in test_results.columns] ,escape=False)
        print('Latex table (test): ',tex_test_table_name)
         
        
        
        
        
        
    # ============================ #
    # Classification
    # ========================== #  
    if do_class:
        
        # Output directory 
        out_dir = out_dir_class            
        out_dir_plots = '{0}/plots'.format(out_dir)
        if not os.path.exists(out_dir_plots):
            os.makedirs(out_dir_plots)  
           
        # Set class according to the chosen bins
        y_trainvalid = set_class(y_trainvalid, bins)
        y_trainvalid_pca = y_trainvalid
        y_test = set_class(y_test, bins)
            
        if do_explore:
            plt.figure()
            y_trainvalid.plot.hist(bins= bins)
            plt.title('Train')
            plt.figure()
            y_test.plot.hist(bins= bins)
            plt.title('Test')

        # Resample train classes
        adasyn = ADASYN()
        X_trainvalid, y_trainvalid = adasyn.fit_resample(X_trainvalid, y_trainvalid)
        X_trainvalid_pca, y_trainvalid_pca = adasyn.fit_resample(X_trainvalid_pca, y_trainvalid_pca)
                
        # Train models
        grid_search_results = {}
        test_results = pd.DataFrame(columns = ['Model','Precision','Recall','F1-Score'])
        test_results_pca = pd.DataFrame(columns = test_results.columns)

        # Train models
        for row, model in enumerate(class_models):
            print('Model: ', model)
    
            rf, parameters, model_title = get_classification_model(model,f_maxsel,random_state = random_state, is_pca= False)
            rf_pca, parameters_pca, model_title_pca = get_classification_model(model, f_maxsel, random_state = random_state, is_pca = True)
            labels = list(range(0, len(bins)-1))
            
            # GS
            grid_search_results = {}
            
            # Load GS results
            gs_file = '{0}/grid_search_results_{1}_pca_{2}.pickle'.format(out_dir_class, model, pca_perc) 
            if not recreate and os.path.exists(gs_file):
                clf = pickle.load(open(gs_file,'rb')) 
                print('Loaded grid search results from: {0}'.format(gs_file))
                
                rf = clf[model].best_estimator_
                rf_pca = clf[model+'_PCA'].best_estimator_
            # Do GS
            else:
                class_scorer = make_scorer(f1_score, average='macro')
                
                # No PCA
                clf, rf = grid_search(rf, parameters, X_trainvalid, y_trainvalid, score=class_scorer)
                grid_search_results[model] = clf
                
                # PCA
                clf_pca, rf_pca = grid_search(rf_pca, parameters_pca, X_trainvalid_pca, y_trainvalid_pca, score=class_scorer)
                grid_search_results[model+'_PCA'] = clf_pca
                
                grid_search_results['params_grid'] = parameters
                
                # Save GS results
                #with open(gs_file, 'wb') as handle:
                #    pickle.dump(grid_search_results, handle, protocol=4)
                #    print('Wrote grid search result to: {0}'.format(gs_file))
            
            # Best params
            p = rf.get_params()
            p1 = rf_pca.get_params()   
    
            # Prediction of the tuned model
            train_report, test_report = get_classification_predictions(X_trainvalid, y_trainvalid, X_test, y_test, rf, 
                                                                       train_results, test_results, model_title, row, labels, out_dir = out_dir_plots)
            # PCA
            train_report_pca, test_report_pca = get_classification_predictions(X_trainvalid_pca, y_trainvalid_pca, X_test_pca, y_test, rf_pca, 
                                                                                       train_results_pca, test_results_pca, model_title, row, labels, 
                                                                                       out_dir = out_dir_plots, is_pca = True)

        # Save the table for all models in latex
        test_results = test_results.merge(test_results_pca, on='Model', suffixes=('',' (PCA)'))
        tex_table_name = out_dir_reg + '/class_table.tex'
        test_results.to_latex(tex_table_name, columns = test_results.columns,
                                  index = False, 
                                  float_format = lambda x: '%.2f' % x, label = 'table:iri_class_pred',  
                                  header=[ format_col(col) for col in test_results.columns],
                                  escape=False)
          
            

       

        