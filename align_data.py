"""
@author: Milena Bajic (DTU Compute)
"""

import sys, os, glob, time
import argparse, json
import psycopg2
import pandas as pd
from utils.data_loaders import *
from utils.plotting import *
from utils.matching import *
from utils.analysis import *
import datetime
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks, argrelmin, argrelextrema, find_peaks_cwt, peak_widths
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
#=================================#
# SETTINGS
#=================================#
# Script arguments
parser = argparse.ArgumentParser(description='Please provide command line arguments.')

parser.add_argument('--route', help='Process all trips on this route, given in json file. This needs to be set to find the DRD file.')
parser.add_argument('--trip', type=int, help='Process those trips. If the route argument is passed in addition, the results for all trips will be in the directory for the selected route. Else, they will be in the unknown_route directory. If the trips are not passed, the trips in the json file for the selected route will be processed.')
parser.add_argument('--is_p79', action='store_true', help = 'If this is p79 data, pass true.')
parser.add_argument('--is_aran', action='store_true', help = 'If this is aran data, pass true.')
parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--step', type=int, default=10)
parser.add_argument('--out_dir', default= "data", help='Output directory.')
parser.add_argument('--recreate', action="store_true", help = 'Recreate files, even if present. If False and the files are present, the data will be loaded from them.')
parser.add_argument('--only_load_results', action='store_true', help='Only load files with results.')
parser.add_argument('--load_add_sensors', action='store_true', help = 'Load additional sensors.') 
parser.add_argument('--predict_mode', action="store_true", help = 'Run in prediction mode. P79 is not needed, GM data will be split into windows.')

    
# Parse arguments
args = parser.parse_args()
trip = args.trip
route = args.route
is_p79 = args.is_p79
is_aran = args.is_aran
json_file = args.json
out_dir_base = args.out_dir
recreate = args.recreate
only_load_results = args.only_load_results
window_size = args.window_size
step = args.step
predict_mode = args.predict_mode
load_add_sensors = args.load_add_sensors

#only_load_results = True
#trips = [7792]      
#is_p79 = True
#route = 'CPH1_VH'
#=================================#        
# Load json file with route info
with open(json_file, "r") as f:
    route_data = json.load(f)
   #print(route_data)

#  # Use the user passed trip, the ones in json file for the route or the default one
if trip:
    GM_trips = [trip]
elif (not trip and route):             
    # Load route data
    GM_trips =  route_data[route]['GM_trips']
# Default trip
else:
    GM_trips = [7792] 
    
    
# Try to find a route if only trip given
if not route:
    # Try to find it from json file and the given trip
    for route_cand in route_data.keys():
        if GM_trips[0] in route_data[route_cand]['GM_trips']:
            route = route_cand
            break
    # If not found in the json file, set to default name if predict mode
    if not route and predict_mode:
        route = 'Copenhagen'
        print('Route not found. Setting to default name: ', route)    
    elif not route:
        print('Please pass the route or add it into the json file. The route is required unless in predict_mode.')
        sys.exit(0)
        
 # DRD trips 
if is_p79:
    DRD_trips =  route_data[route]['P79_trips']
elif is_aran:
    DRD_trips =  route_data[route]['ARAN_trips']     
elif not predict_mode:
    print('Set either p79 or aran to True to do alignement.')
    sys.exit(0)
    
#==================================#    
#===== INPUT/OUTPUT DIRECTORY =====#
if load_add_sensors:
    GM_in_dir = '{0}/GM_processesed_data/{1}_add_sensors/passes'.format(out_dir_base, route) 
else:
    GM_in_dir = '{0}/GM_processesed_data/{1}/passes'.format(out_dir_base, route) 
    
if is_p79:
    DRD_in_dir = '{0}/P79_processesed_data/{1}'.format(out_dir_base, route)
    out_dir = '{0}/aligned_GM_p79_data_window-{2}-step-{3}/{1}'.format(out_dir_base, route, window_size, step)
elif is_aran:
    DRD_in_dir = '{0}/ARAN_processesed_data/{1}'.format(out_dir_base, route)
    out_dir = '{0}/aligned_GM_ARAN_data_window-{2}-step-{3}/{1}'.format(out_dir_base, route, window_size, step)
elif predict_mode:
    out_dir = '{0}/predict_mode_GM_data_window-{2}-step-{3}/{1}'.format(out_dir_base, route, window_size, step)
    
# Create output directory
if load_add_sensors:
    out_dir = '{0}_add_sensors'.format(out_dir)
        
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
 
# Only load alignement results and exit
if only_load_results:
    filenames = []
    for GM_trip in GM_trips: 
        filenames.extend(glob.glob('{0}/*{1}*.pickle'.format(out_dir, GM_trip)))
    GM_data = {}
    for filename in filenames:
        print('Loading: ',filename)
        GM_data[filename.split('/')[-1]] = pd.read_pickle(filename)
        
    #g = GM_data['aligned_GM-7792_DRD-d6b1cf27-41ed-43b6-8050-2068ef941a0aDRD-d6b1cf27-41ed-43b6-8050-2068ef941a0a_GM-7792_pass-1.pickle']  # remove later
    sys.exit(0)
 
print('p79 data? ', is_p79)
print('Aran data? ', is_aran)
print('Additional sensors? ', load_add_sensors)
print('Trip: ', trip)
print('Input directory (GM): ', GM_in_dir)
print('Output directory: ', out_dir)
time.sleep(3)
    
#=================================#    
#======== PROCESS DRD TRIPS ======#
# Load DRD data
if not predict_mode:
    drd_trip  = DRD_trips[0]
    drd_filename = glob.glob('{0}/*{1}*.pickle'.format(DRD_in_dir,drd_trip))[0]
    print('Loading ',drd_filename)
    DRD_data = pd.read_pickle(drd_filename)
    
    # DRD windows
    if is_aran:
        DRD_windows  = get_segments(DRD_data, window_size = window_size, step = step, is_aran = True)
        
        # Compute DI and DI_red(new cols are added)
        compute_di_aran(DRD_windows)
        
    else:
        DRD_windows = get_windows_df(DRD_data, window_size, step)
    


#=================================#    
#======== PROCESS GM TRIPS ======#
all_aligned = {}
number_of_segments = 0

for GM_trip in GM_trips: 
    
     # Load car data from db or file
    print('\nProcessing GM trip: ',GM_trip)
   
    all_aligned[GM_trip] = {}
    # Find available and validated passes
    GM_trip_passes_avail = glob.glob('{0}/interpolated_GM_trip_{1}_pass_*_interpolated.pickle'.format(GM_in_dir, GM_trip))
    GM_trip_passes_avail.sort()
    GM_trip_passes_avail_counter = [int(f.split('/')[-1].split('_')[-2]) for f in GM_trip_passes_avail]
    print('Available passes: ',GM_trip_passes_avail_counter)
    
    try:
        GM_trip_passes_val_counter = route_data[route]['GM_validated_passes'][str(GM_trip)]
        print('Validated passes: ',GM_trip_passes_val_counter)
    except:
        GM_trip_passes_val_counter = []
        print('No info about validated passes for trip {0} found in json file.'.format(GM_trip))
    
    GM_trip_passes_avail_val_counter = set(GM_trip_passes_val_counter).intersection( set(GM_trip_passes_avail_counter) )
    print('Will use passes: ', GM_trip_passes_avail_val_counter)
    
    GM_trip_passes_val_filenames =  []
    for GM_pass_counter in GM_trip_passes_avail_val_counter:
        filename = glob.glob('{0}/interpolated_GM_trip_{1}_pass_{2}_interpolated.pickle'.format(GM_in_dir, GM_trip, GM_pass_counter))[0]
        GM_trip_passes_val_filenames.append(filename)
    
    print('Will use filenames: ', GM_trip_passes_val_filenames)
    
    # Load a validated GM pass
    # ==================== #
    for file in GM_trip_passes_val_filenames:
        print('Loading: ',file)
        
        GM_trip_thispass_data = pd.read_pickle(file)
        GM_pass_i = int(file.split('/')[-1].split('_')[-2])
        GM_trip_thispass_data = GM_trip_thispass_data.drop(['Date', 'Time', 'index', 'index_diff0','GPS_dt'],axis=1)
        GM_trip_thispass_data.reset_index(inplace = True, drop = True)
        
        if 0 in GM_trip_thispass_data.columns:
            GM_trip_thispass_data.drop([0], axis=1, inplace=True)
            
        if predict_mode:
            GM_windows  = get_windows_GM(GM_trip_thispass_data)
            out_filename = '{0}/GM_trip-{1}_pass-{2}_windows.pickle'.format(out_dir, GM_trip, GM_pass_i)
            GM_windows.to_pickle(out_filename)
            print('Saved: ',out_filename)
            
        else:
            # DRD copy
            DRD_windows_copy = DRD_windows.copy()
            DRD_windows_copy.reset_index(inplace = True, drop = True)
    
            
            # Align with DRD
            # ==================== #
            file_suff = 'DRD-{0}_GM-{1}_pass-{2}'.format(drd_trip, GM_trip, GM_pass_i)
            
            aligned_segments, no, nf = align_GM_DRD(GM = GM_trip_thispass_data, iri_data_segments = DRD_windows_copy, GM_trip_id = GM_trip, DRD_trip_id = drd_trip, window_size = window_size, step = step, 
                                                    out_dir = out_dir, file_suff = file_suff)
    
            number_of_segments = number_of_segments + aligned_segments.shape[0]
            #all_aligned[GM_trip][file] = aligned_segments
        #sys.exit(0) #remove!

if not predict_mode: 
    print('Number of matched segments: ', number_of_segments)
    
    
    
    
    
    
    
    