"""
@author: Milena Bajic (DTU Compute)
"""

import sys, os, argparse
import pandas as pd
from utils.data_loaders import *
from utils.plotting import *
from utils.matching import *
import json

#=================================#
# SETTINGS
#=================================#
# Script arguments
parser = argparse.ArgumentParser(description='Please provide command line arguments.')

parser.add_argument('--route', default= "CPH1_VH", help='Process all trips on this route, given in json file.')
parser.add_argument('--trips', nargs='+', type=int, help='Process those trips. If not passed, the trips in the json file for the selected route will be processed.')
parser.add_argument('--p79', action='store_true', help = 'If this is p79 data, pass true.')
parser.add_argument('--aran', action='store_true', help = 'If this is aran data, pass true.')
parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--out_dir', default= "data", help='Output directory.')
parser.add_argument('--recreate', action='store_true', help = 'Recreate files, even if present. If False and files are present, the data will be loaded from them.') 

# Parse arguments
args = parser.parse_args()
route = args.route
trips = args.trips
is_p79 = args.p79
is_aran = args.aran
json_file = args.json
out_dir = args.out_dir
recreate = args.recreate
#=================================
# P79 or ARAN?
if not is_p79 and not is_aran:
    print('No data type set. Choose p79 or aran.')
    sys.exit(0)
    
if is_p79 and is_aran:
    print('Choose either p79 or aran, not both at the same time.')
    sys.exit(0)
    
# Create output directory for this route
if is_p79:
    out_dir_route = '{0}/P79_processesed_data/{1}'.format(out_dir, route)
elif is_aran:
    out_dir_route = '{0}/ARAN_processesed_data/{1}'.format(out_dir, route)
if not os.path.exists(out_dir_route):
    os.makedirs(out_dir_route)

# Create putput directory for route plots
out_dir_plots = '{0}/plots'.format(out_dir_route)
if not os.path.exists(out_dir_plots):
    os.makedirs(out_dir_plots)
    
    
 # Use all selected trips or the ones in json file 
if trips: 
    trips_thisroute = trips
else:
    # Load json file
    with open(json_file, "r") as f:
        route_data = json.load(f)
    if is_p79:
        trips_thisroute =  route_data[route]['P79_trips']
    elif is_aran:
        trips_thisroute =  route_data[route]['ARAN_trips']     
    
    
if not trips_thisroute:
    print('No trips set.')
    sys.exit(0)
       
# Process trips
for trip in trips_thisroute:
    
    # Load data
    DRD_data, iri, DRD_trips = load_DRD_data(trip, is_p79 = is_p79, is_ARAN = is_aran) 
       
    # Map match 
    file_suff =  'P79_route-{0}_taskid-{1}_full'.format(route, trip)
    full_filename = '{0}/map_matched_data{1}.pickle'.format(out_dir_route, file_suff)
    print(full_filename)
    
    if os.path.exists(full_filename):
        map_matched_data  = pd.read_pickle(full_filename)
   
    else:
        if is_aran:
            DRD_data.dropna(subset=['lat','lon'], inplace=True) 
            map_matched_data = map_match_gps_data(gps_data = DRD_data, is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff, recreate = recreate)
        else:
            map_matched_data = map_match_gps_data(gps_data = iri, is_GM = False, out_dir = out_dir_route , out_file_suff = file_suff, recreate = recreate)
        
    plot_geolocation(map_matched_data['lon_map'],  map_matched_data['lat_map'], name = 'DRD_{0}_GPS_mapmatched_gpspoints'.format(trip), out_dir = out_dir_plots, plot_firstlast = 1000, recreate = True)