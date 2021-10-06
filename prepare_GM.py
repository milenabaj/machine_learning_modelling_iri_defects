"""
@author: Milena Bajic (DTU Compute)
"""

import sys, os, glob
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

parser.add_argument('--route', help='Process all trips on this route, given in json file.')
parser.add_argument('--trip', type=int, help='Process this trip.')
parser.add_argument('--skip_interp', action='store_true', help='Do not interpolate.')
parser.add_argument('--only_load_pass_plots', action='store_true', help='Only load GM pass plots.')
parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--out_dir', default= "data/GM_processesed_data", help='Output directory.')
parser.add_argument('--recreate', action="store_true", help = 'Recreate all files. If recreate is false and the files are present, the data will be loaded from files.')
parser.add_argument('--recreate_interp', action="store_true", help = 'Recreate only interpolation files. If recreate is false and the files are present, the data will be loaded from files.')
parser.add_argument('--load_add_sensors', action='store_true', help = 'Load additional sensors.') 
parser.add_argument('--dev_mode',  action="store_true", help = 'Development mode. A small portion of data will be loaded and processes.')   

# Parse arguments
args = parser.parse_args()
trip = args.trip
route = args.route
json_file = args.json
out_dir_base = args.out_dir
recreate = args.recreate
recreate_interp = args.recreate_interp
if recreate==True:
    recreate_interp = True
skip_interpolation = args.skip_interp
plot_html_map = True
only_load_pass_plots = args.only_load_pass_plots
load_add_sensors = args.load_add_sensors
dev_mode = args.dev_mode
dev_nrows = 10000 

print('Dev mode: ',dev_mode)
#=================================#        
# Load json file with route info
with open(json_file, "r") as f:
    route_data = json.load(f)
    
#  # Use the user passed trip, the ones in json file for the route or the default one
if trip:
    GM_trips_thisroute = [trip]
elif (not trip and route):             
    # Load route data
    GM_trips_thisroute =  route_data[route]['GM_trips']
# Default trip
else:
    GM_trips_thisroute = [7792] 
    
    
# Try to find a route if only trip given
if not route:
    # Try to find it from json file and the given trip
    for route_cand in route_data.keys():
        if GM_trips_thisroute[0] in route_data[route_cand]['GM_trips']:
            route = route_cand
            break
    # If not found in the json file, set to default name
    if not route:
        route = 'Copenhagen'
        print('Route not found. Setting to default name: ', route)    

  
# Additional sensors to load
add_sensors = []
if load_add_sensors:
    steering_sensors = ['obd.strg_pos', 'obd.strg_acc','obd.strg_ang'] 
    wheel_pressure_sensors =  ['obd.whl_prs_rr', 'obd.whl_prs_rl','obd.whl_prs_fr','obd.whl_prs_fl'] 
    other_sensors = ['obd.acc_yaw','obd.trac_cons']
    add_sensors = steering_sensors + wheel_pressure_sensors + other_sensors
 
#==================================#    
#===== INPUT/OUTPUT DIRECTORY =====#

# Create output directory for this route if it does not exist 
out_dir_route = '{0}/{1}'.format(out_dir_base, route)
if load_add_sensors:
   out_dir_route = '{0}_add_sensors'.format(out_dir_route)
   
if not os.path.exists(out_dir_route):
    os.makedirs(out_dir_route)

# Create putput directory for route plots
out_dir_plots = '{0}/plots'.format(out_dir_route)
if not os.path.exists(out_dir_plots):
    os.makedirs(out_dir_plots)

# Create putput directory for merged plots
out_dir_plots_merged = '{0}/plots_merged'.format(out_dir_route)
if not os.path.exists(out_dir_plots_merged):
    os.makedirs(out_dir_plots_merged)
    
# Create putput directory for passes
out_dir_passes = '{0}/passes'.format(out_dir_route)
if not os.path.exists(out_dir_passes):
    os.makedirs(out_dir_passes)
   
     
#=================================#    
#======== PROCESS TRIPS ==========#
for GM_trip in GM_trips_thisroute:    
     # Load car data from db or file
    print('\nProcessing GM trip: ',GM_trip)
            
    #============== Only load options ===============#
    # Only load matplotlib plots to make html map plots (use this to make map plots)
    if only_load_pass_plots:
        pattern = '{0}/GM_trip_*_pass_*.pickle'.format(out_dir_plots)
        for name in glob.glob(pattern):
            #if os.path.exists(name.replace('.png','_printout.png')):
            #    continue
            
            filename = name.split(out_dir_plots+'/')[1]
            print('Using: ',name)
            plot_geolocation(full_filename = name, out_dir = out_dir_plots, plot_html_map = True)
                   
        # Create a merged pdf file with plots 
        pattern = '{0}/GM_trip*.png'.format(out_dir_plots)
        files = glob.glob(pattern)        
        files = [f for f in files if '_map.png' not in f]
        files.sort(key = lambda f: (int(f.split('/')[-1].split('_')[2]), sort2(f), sort3(f)) )
        
        from PIL import Image, ImageFont, ImageDraw 
        imagelist = [ ]
        for file in files:
           name = file.split('/')[-1].replace('.png','')
           img =  Image.open(file).convert('RGB')
           draw = ImageDraw.Draw(img)
           if 'pass' in name:
               font = ImageFont.truetype(r'/Library/Fonts/Arial Unicode.ttf',40)
               draw.text((50, 20),name, 'black', font) 
           elif 'minima' in name:
               font = ImageFont.truetype(r'/Library/Fonts/Arial Unicode.ttf',10)
               draw.text((50, 0),name, 'black', font)  
           else:
               font = ImageFont.truetype(r'/Library/Fonts/Arial Unicode.ttf',10)
               draw.text((70, 20),name, 'black', font) 
           imagelist.append(img)
  
        out_filename = '{0}/GM_route_{1}_merged_plots.pdf'.format(out_dir_plots_merged, route)
        imagelist[0].save(out_filename,save_all=True, append_images=imagelist[1:])
        print('Merge images saved as: ', out_filename)
        
        continue # skip the rest of the code and go to the next trip
        
     
     #============== Load the trip ===============#
    filename = '{0}/GM_db_meas_data_{1}.pickle'.format(out_dir_route, GM_trip)
    if os.path.exists(filename) and not recreate:
        print('Loading GM trip from file: ',GM_trip)
        GM_data = pd.read_pickle(filename)
        if dev_mode:
            GM_data=GM_data.head(dev_nrows)
        #GM_trip_info  = pd.read_pickle('{0}/GM_db_trips_info.pickle'.format(out_dir_route))
    else:     
        print('Loading GM trip from db: ',GM_trip)
        GM_data, GM_trip_info = load_GM_data(GM_trip, out_dir = out_dir_route, add_sensors = add_sensors, load_nrows=dev_nrows) 
    

    #============== Map match the trip===============#
    print('Starting map matching')
    
    # GPS dataframe
    gps = GM_data[GM_data['T']=='track.pos']  
    GM_data = GM_data[GM_data['T']!='track.pos']  
    
    map_filename = 'mapmatched_gpspoints_fulltrip_{0}'.format(GM_trip)
    gps_mapmatched = map_match_gps_data(gps_data = gps, is_GM = True, out_dir = out_dir_route, out_file_suff = '_GM_{0}'.format(GM_trip), recreate = recreate)
   
    # Plot map matched
    plot_filename = '{0}/GM_trip_{1}_mapmatched_gpspoints_fulltrip.png'.format(out_dir_plots, GM_trip)
    ax = plt.scatter(gps_mapmatched['lon_map'], gps_mapmatched['lat_map'], s=5)
    fig = ax.get_figure()
    fig.suptitle('GM trip {0} '.format(GM_trip))
    fig.savefig(plot_filename.replace('distance0','indexdiff0'))
    print('Wrote to: ',plot_filename)

    #a =  (gps_mapmatched.iloc[200]['lat_map'], gps_mapmatched.iloc[200]['lon_map'])
    #b = (gps_mapmatched.iloc[3000]['lat_map'], gps_mapmatched.iloc[210]['lon_map'])
    #d = distance.euclidean(a, b)
    
    #============== Remove outliers (DBScan) ===============#
    print('Removing outliers')
    model = DBSCAN(eps=0.01, min_samples=20).fit(gps_mapmatched[['lon_map','lat_map']])
    gps_mapmatched['label'] = model.labels_
    
    # Find and plot clusters
    ax = plt.scatter(gps_mapmatched['lon_map'], gps_mapmatched['lat_map'], s=5, c=gps_mapmatched['label'])
    fig.suptitle('GM trip {0}: Clusters'.format(GM_trip))
    fig = ax.get_figure()
    fig.savefig(plot_filename.replace('_mapmatched','_wtr1stpoint_mapmatched').replace('.png','_clusters.png'))
    
    # Check which labels to keep
    nc = gps_mapmatched['label'].value_counts(normalize=True, sort=True).to_dict()
    keep_labels = []
    for l, count in nc.items():
        if count>0.01:
            keep_labels.append(l)
    print(keep_labels)


    # Remove outliers
    gps_mapmatched = gps_mapmatched[gps_mapmatched['label'].isin(keep_labels)]
    gps_mapmatched.reset_index(drop=True, inplace=True)
    ax = plt.scatter(gps_mapmatched['lon_map'], gps_mapmatched['lat_map'], s=5)
    fig.suptitle('GM trip {0}: Removed outliers'.format(GM_trip))
    fig = ax.get_figure()
    fig.savefig(plot_filename.replace('.png','_removed_outliers.png'))
    print('Wrote to: ',plot_filename)
    
    # Plot
    #plot_geolocation(gps_mapmatched['lon_map'], gps_mapmatched['lat_map'], name= map_filename,out_dir = out_dir_plots, plot_firstlast = 10, do_open = False)
    #plot_geolocation(gps_result['lon_map'][0:1000], gps_result['lat_map'][0:1000], name = 'GM_{0}_GPS_mapmatched_points_start'.format(GM_trip), out_dir = our_dir_plots, plot_firstlast = 5)
 
    #============== Split the trip into passes ===============#
    #GM_int_data = GM_int_data.iloc[:50000]  
    print('Splitting into passes')
    
    gps_mapmatched.reset_index(drop=True, inplace=True)
    gps_mapmatched['index'] = gps_mapmatched.index
    
    # The first point
    lat0 =  gps_mapmatched.iloc[0]['lat_map']
    lon0 =  gps_mapmatched.iloc[0]['lon_map']
    t0 =  gps_mapmatched.iloc[0]['TS_or_Distance']
    i0 =  gps_mapmatched.iloc[0]['index']
    
    # Compute differences wtr to the first point
    gps_mapmatched['distance0'] =  gps_mapmatched.apply(lambda row: haversine_distance(lat0, row['lat_map'], lon0, row['lon_map']), axis=1)
    gps_mapmatched['time_diff0'] =  gps_mapmatched.apply(lambda row: pd.Timedelta(row['TS_or_Distance'] - t0), axis=1)
    gps_mapmatched['time_diff0'] = gps_mapmatched['time_diff0'].apply(lambda row: row.seconds/60)
    gps_mapmatched['index_diff0'] =  gps_mapmatched.apply(lambda row: row['index'] - i0, axis=1)
  
    # Fit index difference vs distance
    rmse = {}
    for d in list(range(5,70,5)):
        model = polynomial_model(degree=d)
        model.fit(gps_mapmatched['index_diff0'].to_numpy().reshape(-1, 1) , gps_mapmatched['distance0'].to_numpy().reshape(-1, 1))
        pred = model.predict(gps_mapmatched['index_diff0'].to_numpy().reshape(-1, 1))
        rmse[d] = mean_squared_error(gps_mapmatched['distance0'].to_numpy().reshape(-1, 1), pred)
     
    # Best fit
    best_d = min(rmse, key = rmse.get) 
    model = polynomial_model(degree=best_d)
    model.fit(gps_mapmatched['index_diff0'].to_numpy().reshape(-1, 1) , gps_mapmatched['distance0'].to_numpy().reshape(-1, 1))
    pred = model.predict(gps_mapmatched['index_diff0'].to_numpy().reshape(-1, 1))
    
    # Find valleys
    pred_inv = -1*pred.reshape(-1)
    p = pred.max()/8
    minima_indices_cand = find_peaks(pred_inv, prominence=p,distance=500)[0]
    #w = peak_widths(pred.reshape(-1), peaks)
    
    # Find array with minima
    #o = int(gps_mapmatched['distance0'].shape[0]/20)
    #minima_indices_cand = list(argrelmin(pred, order = o)[0]) # or pred
    minima_indices = []
    for i in minima_indices_cand:
        distance0 = gps_mapmatched[gps_mapmatched['index_diff0'] == i]['distance0'].values[0]
        if (i>500):  #remove if those are first points when car is setting off
            minima_indices.append(i)
    print('Minima found at: ',minima_indices)
 
    # Plot distance difference wtr to time and save
    plot_filename = '{0}/GM_trip_{1}_distance0_wtr_time_mapmatched_gpspoints_fulltrip.png'.format(out_dir_plots, GM_trip)
    ax = gps_mapmatched.plot('time_diff0','distance0', kind='scatter', s=3)
    fig = ax.get_figure()
    ax.set_title('GM trip: {0}'.format(GM_trip))
    #fig.savefig(plot_filename)
    
    # Plot index wtr to time 
    ax = gps_mapmatched.plot('index_diff0','distance0', kind='scatter', s=3, label = 'Data')
    ax.plot(gps_mapmatched['index_diff0'].to_numpy(), pred, c= 'red', label='Fitted function')
    
    # Add minima to the plot (new passes) and save the figure
    for i in minima_indices:
        ax.axvline(x=i, c='b')
    ax.legend(loc="lower right",frameon=False)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(plot_filename.replace('fulltrip','fulltrip_minima'))
    print('Wrote to: ',plot_filename)
            
    # List with borders of different passes
    lower_borders = [0]+minima_indices
    upper_borders = minima_indices + [gps_mapmatched.shape[0]]
    borders = list(zip(lower_borders, upper_borders))
    print(borders)
    
    
    # ============== Process different passes ==============#
    for i,(low,up) in enumerate(borders):
        # pass start gps, compute distance and take end from at least 100m from the start
        print('Processing trip: {0}, trip: {1}'.format(GM_trip, i))
        
        # if super small pass, ignore
        if up-low<500:
            continue
        
        #upb = up-200
        upb = up-1
        
        # Df for this pass
        gps_car_pass = gps_mapmatched[gps_mapmatched['index_diff0'].between(low,upb)]
        gps_car_pass.drop(['distance0','time_diff0','label'],axis=1, inplace=True)
        gps_car_pass.reset_index(drop=True, inplace=True)
        s = gps_car_pass.shape
        print('pass: {0}, borders: {1}-{2}, shape: {3}'.format(i, low, upb, s))
       
        
        # Plot the pass
        fig = plot_geolocation(gps_car_pass['lon_map'], gps_car_pass['lat_map'], name = 'GM_trip_{0}_pass_{1}_GPS_mapmatched_gpspoints'.format(GM_trip, i), out_dir = out_dir_plots, plot_firstlast = 10, 
                               plot_html_map = plot_html_map, title = 'GM trip: {0}, pass:{1}'.format(GM_trip, i))
        
    
        # Find full GM data for this pass
        t0 = gps_car_pass['TS_or_Distance'].iloc[0]
        tf = gps_car_pass['TS_or_Distance'].iloc[-1]
        GM_pass = GM_data[GM_data['TS_or_Distance'].between(t0, tf)]
        
        # Merge map matched GPS with the full dataframe 
        GM_pass_full_data = pd.concat([GM_pass, gps_car_pass.drop(['Date','Time'],axis=1) ], ignore_index=True)
        GM_pass_full_data.sort_values(by='TS_or_Distance',ascending=True, inplace=True)
        
        # Remove not needed columns
        #GM_pass_full_data.drop([0,'Date','Time'],axis=1,inplace=True)
        
        # Set Message to nan if from GPS 
        GM_pass_full_data['Message'].mask(GM_pass_full_data['T']=='track.pos',inplace=True)
        GM_pass_full_data.reset_index(drop=True, inplace=True)
       
        # Save the pass df
        #out_filename = '{0}/GM_trip_{1}_pass_{2}_mapmatched_gpspoints.pickle'.format(out_dir_passes, GM_trip, i)
        #gps_car_pass.to_pickle(out_filename)
        
        # Interpolate the pass df
        #GM_map_matched_data = GM_map_matched_data.iloc[8000:9000]

        if not skip_interpolation:
             print('Interpolating.......')
             # Out filename
             inter_filename = 'GM_trip_{0}_pass_{1}_interpolated'.format(GM_trip, i)
             
             # Interpolate
             GM_int_data, gps = interpolate_trip(all_sensor_data = GM_pass_full_data, out_dir = out_dir_passes, add_sensors = add_sensors, file_suff = inter_filename, recreate = recreate_interp) 
             
             # Filter
             GM_int_data = GM_int_data[GM_int_data['GPS_dt']<5]
             
             # Plot
             #GM_int_data['GPS_dt'].describe()
             plot_geolocation(gps['lon_map'], gps['lat_map'], name = 'GM_trip_{0}_pass_{1}_GPS_mapmatched'.format(GM_trip, i), out_dir = out_dir_plots, plot_firstlast = 10, 
                              plot_html_map = plot_html_map, title = 'GM trip: {0}, pass:{1}'.format(GM_trip, i))
             plot_geolocation(GM_int_data['lon_int'][::300], GM_int_data['lat_int'][::300], name = 'GM_trip_{0}_pass_{1}_GPS_interpolated_300th'.format(GM_trip, i), out_dir = out_dir_plots, plot_firstlast = 10, 
                               plot_html_map = plot_html_map, title = 'GM trip: {0}, pass:{1}, interpolated'.format(GM_trip, i))

    # Close all figures
    plt.close('all')



   
    