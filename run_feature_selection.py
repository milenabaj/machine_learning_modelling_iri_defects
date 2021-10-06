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
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import classification_report, plot_confusion_matrix
from utils.analysis import *
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter
from math import ceil, floor
import gc, os, sys, glob, time            

if __name__=='__main__':
    
    # ======================= #
    # Setup
    # ======================= #  

    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    parser.add_argument('--is_p79', action='store_true', help = 'If this is p79 data, pass true.')
    parser.add_argument('--is_aran', action='store_true', help = 'If this is aran data, pass true.')
    parser.add_argument('--aran_target',  default='DI')
    parser.add_argument('--recreate', action="store_true", help = 'Recreate files, even if present. If False and the files are present, the data will be loaded from them.')
    parser.add_argument('--dev', action="store_true", help = 'If developing, set to true, the number of trees will be small.') 
  
    parser.add_argument('--base_dir', default= "data", help='Output directory.')
    parser.add_argument('--suff', default= "", help='Output directory suffix.') 
    parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
    
    # Select what to use 
    parser.add_argument('--routes', action = 'append', help='Process all routes.')
    parser.add_argument('--trip', type=int, help='Process this trip only. By default, used is pass 0 only.') 
    parser.add_argument('--speedacc_feats', action = 'append', help='Consider those features. Use: all_accspeed to use acc in 3-axis and speed or pass a combination of acc_x, acc_y, acc_z and speed.')    
   
    # Load additional sensors
    parser.add_argument('--load_add_sensors', action='store_true', help = 'Load additional sensors.')  
      
    # Parse arguments
    args = parser.parse_args()
    
    is_p79 = args.is_p79
    is_aran = args.is_aran
    aran_target_name = args.aran_target
    iri_target_name = 'IRI_mean'
    recreate = args.recreate
    dev = args.dev
    
    base_dir = args.base_dir
    json_file = args.json
    suff = args.suff
    
    routes = args.routes
    trip = args.trip #10900
    
    load_add_sensors = args.load_add_sensors

    nt = 50
    do_reg_fs = True
    do_class_fs = False
    bins_comb = [[0,0.9,1.2,1.8,2.4,4],[0,0.9,1.2,2.4,4]]
    window_size = 100
    step = 10
    fmax = 30
    
    # ======================= #
    # INPUT SENSORS
    # ======================= #
    use_feats = []
    if not args.speedacc_feats:
        # Default setting
        use_feats = ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z'] # use those for final files
        out_name = 'speed-acc_x-acc_y-acc-z'
    # Else take from arguments
    else:
        out_name = ''
        for feat in args.speedacc_feats:
            if feat=='all_accspeed':
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
    
    # Additional sensors to load
    if load_add_sensors:
        steering_sensors = ['GM.obd.strg_pos.value', 'GM.obd.strg_acc.value','GM.obd.strg_ang.value'] 
        wheel_pressure_sensors =  ['GM.obd.whl_prs_rr.value', 'GM.obd.whl_prs_rl.value','GM.obd.whl_prs_fr.value','GM.obd.whl_prs_fl.value'] 
        other_sensors = ['GM.obd.acc_yaw.value','GM.obd.trac_cons.value']
        add_sensors = steering_sensors + wheel_pressure_sensors + other_sensors 
        use_feats =  use_feats + add_sensors
        out_name = out_name+'-steering-wheelpressure-yaw-traccons'
    
    if out_name.startswith('-'):
        out_name = out_name[1:]
    
    
    
    # ======================= #
    # TRIPS AND ROUTE
    # ======================= #
    #  # Use the user passed trip, the ones in json file for the route or the default one
    if trip:
        GM_trips = [trip]
    elif (not trip and route):             
        # Load route data
        GM_trips =  route_data[route]['GM_trips']
    # Default trip
    else:
        GM_trips = [7792] 
        
    # Load json file with route info
    with open(json_file, "r") as f:
        route_data = json.load(f)
       #print(route_data)


    # Try to find a route if only trip given
    if trip and not routes:
        # Try to find it from json file and the given trip
        for route_cand in route_data.keys():
            if GM_trips[0] in route_data[route_cand]['GM_trips']:
                route = route_cand
                break
        if not route:
            print('Please pass the route or add it into the json file.')
            sys.exit(0)
        routes=[route] 
    # If a specific trip is not required, then load the default routes
    else:
        routes = ['CPH1_HH', 'CPH1_VH']
    combined_route_name = '_'.join(routes)

    # Output file with a list of selected features
    trips_string = '-'.join([str(trip) for trip in GM_trips])
    if is_p79:
        json_feats_file = 'json/selected_features_p79_route-{0}_GM_trip-{1}_sensors-{2}.json'.format(combined_route_name, trips_string, out_name)
    elif is_aran:
        json_feats_file = 'json/selected_features_ARAN_route-{0}_GM_trip-{1}_sensors-{2}.json'.format(combined_route_name, trips_string, out_name)
    
   
    # =========================== #
    # INPUTS AND OUTPUT DIRECTORIES
    # ============================ #
    # Input directory
    in_dirs = []
    
    if is_p79:            
        for route in routes:
            in_dir = '{0}/aligned_GM_p79_data_window-{1}-step-{2}-feature-extraction/{3}'.format(base_dir, window_size, step, route)
           #if args.trip:
            #    in_dir =  in_dir+'/chunks'
            if load_add_sensors:
                in_dir = in_dir + '_add_sensors'
            in_dirs.append(in_dir)
        target_name = iri_target_name
    elif is_aran:
        for route in routes:
            in_dir = '{0}/aligned_GM_ARAN_data_window-{1}-step-{2}-feature-extraction/{3}'.format(base_dir, window_size, step, route) 
            #if args.trip:
             #    in_dir =  in_dir+'/chunks'
            if load_add_sensors:
                in_dir = in_dir + '_add_sensors'
            if aran_target_name!='DI':
                in_dir = in_dir+'_individual_defects'
            in_dirs.append(in_dir)
        target_name = aran_target_name
    else:
        print('Error in input settings! Exiting..')
        sys.exit(0)
        
    # Output directory for FS files
    if args.trip:
        out_dir = in_dir.replace('feature-extraction','feature-selection-{0}{1}'.format(out_name, suff))
    else:
        out_dir = '/'.join(in_dir.split('/')[0:-1])
        out_dir =  out_dir.replace('feature-extraction','feature-selection-{0}{1}'.format(out_name, suff))
        out_dir = '{0}/{1}'.format(out_dir, combined_route_name)
    
    out_dir = out_dir + '_' + target_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
     
    print('Is p79? ', is_p79)
    print('is ARAN? ',is_aran)
    print('Will use input features: ',use_feats)
    print('Output suffix: ',out_name)
    print('Input directory: ',in_dir)
    print('Output directory: ',out_dir)
    time.sleep(3)
    
    # ======================= #
    # Input files
    # ======================= 
    train_dfs = []
    valid_dfs = []
    test_dfs = []
    for in_dir in in_dirs:
        print('Input directory: ',in_dir)
        
        # Train
        train_filename = glob.glob('{0}/train_route*.pickle'.format(in_dir))[0]
        train_df = pd.read_pickle(train_filename)
        
        # Valid
        valid_filename = glob.glob('{0}/valid_route*.pickle'.format(in_dir))[0]
        valid_df = pd.read_pickle(valid_filename)
            
        # Test
        test_filename = glob.glob('{0}/test_route*.pickle'.format(in_dir,))[0]
        test_df = pd.read_pickle(test_filename)
        
        train_dfs.append(train_df)
        valid_dfs.append(valid_df)   
        test_dfs.append(test_df)
        '''
        if not args.trip:
            # Train
            train_filename = glob.glob('{0}/train_route*.pickle'.format(in_dir))[0]
            train_df = pd.read_pickle(train_filename)
            
            # Valid
            valid_filename = glob.glob('{0}/valid_route*.pickle'.format(in_dir))[0]
            valid_df = pd.read_pickle(valid_filename)
                
            # Test
            test_filename = glob.glob('{0}/test_route*.pickle'.format(in_dir))[0]
            test_df = pd.read_pickle(test_filename)
            
            train_dfs.append(train_df)
            valid_dfs.append(valid_df)   
            test_dfs.append(test_df)
        else:
            # 0th pass is taken by default
            filename = glob.glob('{0}/*GM-{1}*_pass-0_chunk_0.pickle'.format(in_dir, trip))[0]
            df = pd.read_pickle(filename)
            
            # Extract and clean
            extract_inner_df(df, use_feats, do_clean_nans=True)    
            
            #s = df.street_name_start.unique()  
            train_streets = ['Vigerslevvej', 'Ålholmvej', 'Grøndals Parkvej', 'Rebildvej',
               'Sallingvej', 'Hulgårdsvej', 'Tomsgårdsvej', 'Tuborgvej',
               'Lyngbyvej', '', 'Helsingørmotorvejen', 'Nordhavnsvej',
               'Strandvænget', 'Kalkbrænderihavnsgade', 'Folke Bernadottes Allé',
               'Grønningen', 'Holbergsgade']
            valid_streets = ['Niels Juels Gade', "Christian D. IV's Bro",'Christians Brygge', 'Bryghusbroen']
            test_streets = ['Kalvebod Brygge','Vasbygade','Sydhavns Plads', 'P. Knudsens Gade', 'Ellebjergvej','Folehaven']
            
            # Train
            train_df = df[df.street_name_start.isin(train_streets)]
            train_df.reset_index(inplace=True, drop=True)
            train_dfs.append(train_df)  
            
            # Valid
            valid_df = df[df.street_name_start.isin(valid_streets)]
            valid_df.reset_index(inplace=True, drop=True)
            valid_dfs.append(valid_df) 
            
            # Test
            test_df = df[df.street_name_start.isin(test_streets)]
            test_df.reset_index(inplace=True, drop=True)
            test_dfs.append(test_df)
        '''
    train_df = pd.concat(train_dfs, ignore_index=True)
    non_GM_cols = [col for col in  train_df.columns if not col.startswith('GM.')]
    clean_nans(train_df, exclude_cols=non_GM_cols)
    train_df.reset_index(inplace=True, drop=True)
    train_df[target_name].hist()
        
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    clean_nans(valid_df, exclude_cols=non_GM_cols)
    valid_df.reset_index(inplace=True, drop=True)
    plt.title('Train-Valid')
    valid_df[target_name].hist()
    plt.savefig('train_valid.pdf')
        
    test_df = pd.concat(test_dfs, ignore_index=True)
    clean_nans(test_df, exclude_cols=non_GM_cols)
    test_df.reset_index(inplace=True, drop=True)
    plt.figure()
    test_df[target_name].hist()
    plt.savefig('test.pdf')
         
    # Remove some columns if needed
    to_rem = []
    avail_cols = list(train_df.columns)
    for col in avail_cols:
        if any(x in col for x in ['ECDF Percentile Count', 'ECDF_']):
             to_rem.append(col)  

    train_df.drop(to_rem,axis=1,inplace=True)
    valid_df.drop(to_rem,axis=1,inplace=True)
    test_df.drop(to_rem,axis=1,inplace=True)
   
    # TEST
    #train_df =  train_df.head(100)
    #valid_df =  valid_df.head(100)
    #test_df =  test_df.head(100)
    
    # Select columns to use
    GM_features = [col for col in train_df.columns if col.startswith('GM')]
    
    X_train_fe = train_df[GM_features] 
    y_train =  train_df[target_name]
    
    X_valid_fe = valid_df[GM_features] 
    y_valid =  valid_df[target_name]
    
    X_test_fe = test_df[GM_features] 
    y_test =  test_df[target_name]
    
    
    #SFS requires trainvalid passed together with valid indices
    X_trainvalid_fe = pd.concat([X_train_fe, X_valid_fe], ignore_index=True)
    X_trainvalid_fe.reset_index(inplace=True, drop=True)
    
    y_trainvalid = pd.concat([y_train, y_valid], ignore_index=True)
    y_trainvalid.reset_index(inplace=True, drop=True)
    
    X_valid_indices = np.arange(X_train_fe.shape[0], X_trainvalid_fe.shape[0])

    # ========================== #
    # Regression Feature Selection
    # ========================== #
    if do_reg_fs:
        
        # Do FS on train
        #if args.trip:
            #train_outfile_suff = filename.split('/')[-1].split('.pickle')[0] + '_feature_selection_train'
        #else:
       #     train_outfile_suff = train_filename.split('/')[-1].split('.pickle')[0] + '_feature_selection_train'
        train_outfile_suff = train_filename.split('/')[-1].split('.pickle')[0] + '_feature_selection_train'
        
        valid_ind_file = out_dir + '/' + train_outfile_suff.replace('_feature_selection_train','valid_indices.pickle')
        test_outfile_suff = train_outfile_suff.replace('train','test')
        pickle.dump(X_valid_indices, open(valid_ind_file,'wb'))
         
        X_trainvalid_fs, sel_feature_names = find_optimal_subset(X_trainvalid_fe, y_trainvalid, valid_indices = X_valid_indices, n_trees=nt, reg_model = True, target_name = target_name,
                                                                 out_dir = out_dir, fmax = fmax, outfile_suff = train_outfile_suff, recreate = recreate)
        
        # only select chosen vars on test
        X_test_fs, sel_feature_names =  find_optimal_subset(X_test_fe, y_test,  fmax = fmax, sel_features_names =  sel_feature_names, out_dir = out_dir,  target_name = target_name, 
                                                            outfile_suff = test_outfile_suff, recreate = recreate)
            
        # Write json file
        f =  open(json_feats_file, "w") 
        sel_feat_data = {"features":sel_feature_names}
        json.dump(sel_feat_data, f)    
        f.close()
        n_sel_features = len(sel_feature_names)
        
        print('Selected features are: {0}'.format(sel_feature_names))
        print('Number of selected features is:{0}'.format(n_sel_features))
   
    sys.exit(0)
    
   # ============================ #
    # Classification Feature Selection
    # ========================== #  
    if do_class_fs:
        
        for bins in bins_comb:
        
            y_trainvalid_class = set_class(y_trainvalid, bins)
            y_test_class = set_class(y_test, bins)
            
            # Do FS on train
            X_trainvalid_fs, sel_feature_names = feature_selection(X_trainvalid_fe, y_trainvalid_class, 'train', fs_class_prep_dir, GM_trip_id, target_name, 
                                                           fmax = fmax, bins = bins, n_trees=nt, reg_model = False, recreate = recreate)
            
     
            # only select chosen vars on test
            X_test_fs, sel_feature_names = feature_selection(X_test_fe, y_test_class, 'test', fs_class_prep_dir, GM_trip_id, target_name, 
                                                         sel_feature_names, bins = bins, fmax = fmax, n_trees=nt,  reg_model = False, recreate = recreate)
          
            n_sel_features = len(sel_feature_names)
            print('Selected features are: {0}'.format(sel_feature_names))
            print('Number of selected features is:{0}'.format(n_sel_features))
       
           

        