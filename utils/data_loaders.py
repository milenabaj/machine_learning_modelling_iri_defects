"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2 # pip install psycopg2==2.7.7 or pip install psycopg2==2.7.7
from json import loads
import sys,os, glob
from datetime import datetime
from utils.analysis import *
import pickle
from json import loads

def get_trips_info(task_ids = None):
    
    #conn = psycopg2.connect(database="..", user="..", password="..", host="..", port=..) # replace with your authenticaion details
    
    # Quory
    #quory = 'SELECT * FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT 1000'.format(GM_TaskId)
    if task_ids:
        #quory = 'SELECT * FROM public."Trips" WHERE ("Trips"."Fully_Imported"=\'True\' AND "Trips"."TaskId" IN {0}) ORDER BY "TaskId" ASC'.format(task_ids)
        task_ids=str(tuple(task_ids))
        quory = 'SELECT * FROM public."Trips" WHERE ("Trips"."Fully_Imported"=\'True\' AND "Trips"."TaskId" IN {0}) ORDER BY "TaskId" ASC'.format(task_ids)
    else:
        quory = 'SELECT * FROM public."Trips" WHERE "Trips"."Fully_Imported"=\'True\' ORDER BY "TaskId" ASC'
    
    # Set cursor
    cursor = conn.cursor()
    
    d = pd.read_sql(quory, conn, coerce_float = True) 
    
    # Close the connection
    cursor.close()
    conn.close()  
    
    return d


def get_matching_DRD_info(GM_trip_id):
    
    DRD_info = {}
    if GM_trip_id==4955:
        DRD_trip_id = 'a34887d6-46df-496c-bb82-0c6b205fb199'
    elif GM_trip_id==4957:
        DRD_trip_id = 'a34887d6-46df-496c-bb82-0c6b205fb199'
    elif GM_trip_id==4959:
        DRD_trip_id = '468f5e4c-2977-4785-bea5-d1aca6923435'
    elif GM_trip_id==5683:
        DRD_info['NS'] = 'e9efaba7-a322-4793-a019-9592fb3ee73f' #NS
        DRD_info['SN'] =  'ef5eb740-198e-45c6-a66a-fcbdfb0016fe' #SN
    else:
        print('No DRD trip id set for this GM trip')
        sys.exit(0)

    return DRD_info

def get_matching_ARAN_info(GM_trip_id):
    ARAN_info = {}
    if GM_trip_id==5683:
        ARAN_info['V'] = '974a5c25-ee35-43c6-a2d5-2a486ec6ab0e' #from Brondby up the M3
        ARAN_info['H'] =  '538bd787-06f7-417b-a412-24b0d3caa594' 
    else:
        print('No ARAN trip id set for this GM trip')
        sys.exit(0)
    return ARAN_info

        
def get_GM_passes(GM_trip_id):
    print('Getting GM passes')
    passes = {}
    if GM_trip_id==5683:  
        # up to 136000: not useful
        passes['NS'] = [(136000, 270000)]
        passes['SN'] = [(285000, 457000)]
    else:
        print('No passes info found for this GM trip')
        sys.exit(0)
    return passes

def filter_keys(msg):
    remove_list= ['id', 'start_time_utc', 'end_time_utc','start_position_display',
                  'end_position_display','device','duration','distanceKm','tag', 
                  'personal', '@ts','@uid', '@t','obd.whl_trq_est', '@rec']
    msg = {k : v for k,v in msg.items() if k not in remove_list}
    return msg
 

def extract_string_column(sql_data, col_name = 'message'):
    # if json
    try: 
        sql_data[col_name] = sql_data[col_name].apply(lambda message: loads(message))
    except:
        pass
    keys = sql_data[col_name].iloc[0].keys()
    n_keys =  len(keys)
    for i, key in enumerate(keys):
        print('Key {0}/{1}'.format(i, n_keys))
        sql_data[key] = sql_data[col_name].apply(lambda col_data: col_data[key])
        
    sql_data.drop(columns=[col_name],inplace=True,axis=1)
    return sql_data
    
def check_nans(sql_data, is_aran = False, exclude_cols = []):   
    n_rows = sql_data.shape[0]
    for col in  sql_data.columns:
        if col in exclude_cols:
            continue
        n_nans = sql_data[col].isna().sum()
        n_left = n_rows - n_nans
        print('Number of nans in {0}: {1}/{2}, left: {3}/{2}'.format(col, n_nans, n_rows, n_left ))
    return

def load_GM_data(GM_TaskId, out_dir, all_sensors = False, add_sensors = [], load_nrows = -1):
    
    # Set up connection
     #==============#
    print("\nConnecting to PostgreSQL database to load GM")
    #conn = psycopg2.connect(database="..", user="..", password="..", host="..", port=..) #replace with your authentication details
   
    # Get measurements #
    #========================#
    print('Loading GM measurements from the db')
    #quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T"=\'track.pos\' OR "Measurements"."T"=\'acc.xyz\' OR "Measurements"."T"=\'obd.spd_veh\')) LIMIT 500'.format(GM_TaskId)
    if all_sensors:
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
        else:   
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId)
    else:
        sensors = ['track.pos','acc.xyz','obd.spd_veh']
        if add_sensors:
            sensors = sensors + add_sensors
        sensors = str(tuple(sensors))
        print('Loading: ',sensors)
        #sensors = '(\'track.pos\', \'acc.xyz\')' # works
        #sensors = "('track.pos', 'acc.xyz')" #works
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC LIMIT {2}'.format(GM_TaskId, sensors, load_nrows)  
        else:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId, sensors)     
  

     
    cursor = conn.cursor()
    meas_data = pd.read_sql(quory, conn, coerce_float = True)
    meas_data.reset_index(inplace=True, drop=True)   
    meas_data['Message'] = meas_data.message.apply(lambda msg: filter_keys(loads(msg)))
    meas_data.drop(columns=['message'],inplace=True,axis=1)
    meas_data.reset_index(inplace=True, drop=True)
    meas_data = meas_data[['TS_or_Distance','T', 'lat', 'lon','Message']]
    
    # Extract day and time
    #=================#
    meas_data['Date'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
    meas_data['Time'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())
    meas_data.sort_values(by='Time',inplace=True)
    
  
    # Get GM trips info #
    #=================#
    print('Loading GM trip information')
    quory = 'SELECT * FROM "Trips"'
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn) 
    trips.reset_index(inplace=True, drop=True)   
    
    
    # Close connection
    #==============#
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    
    # Save files
    #==============#
    if all_sensors:
        filename = '{0}/GM_db_meas_data_{1}_allsensors.csv'.format(out_dir, GM_TaskId)
    else:
        filename = '{0}/GM_db_meas_data_{1}.csv'.format(out_dir, GM_TaskId)
        
    #meas_data.to_csv(filename)
    meas_data.to_pickle(filename.replace('.csv','.pickle'))
    
    #meas_data.to_csv('{0}/GM_db_trips_info.csv'.format(out_dir, GM_TaskId))
    meas_data.to_pickle('{0}/GM_db_trips_info.pickle'.format(out_dir, GM_TaskId))
    
    return meas_data, trips
 
        
def load_DRD_data(DRD_trip, is_p79 = False, is_ARAN = False):
       
    # Set up connection
    print("\nConnecting to PostgreSQL database to load the DRD data")
    #conn = psycopg2.connect(database="..", user="..", password="..", host="..", port=..) #replace with your authentication detals

    
    # Execute quory: get sensor data
    print('Selecting data')
    if is_ARAN:
        quory = 'SELECT * FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    elif is_p79:
        #quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE ("FK_Trip"=\'{0}\' AND "lat"<={1} AND "lat">={2} AND "lon"<={3} AND "lon">={4}) ORDER BY "TS_or_Distance" ASC'.format(DRD_trip, lat_max, lat_min, lon_max, lon_min)
        quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    else:
        print('Set either p79 or ARAN to true. Other vehicle trips not implemented yet.')
        sys.exit(0)
    
    # Load and sort data
    cursor = conn.cursor()
    sql_data = pd.read_sql(quory, conn, coerce_float = True)

    # Sort also in pandas after conversion to float
    sql_data.TS_or_Distance = sql_data.TS_or_Distance.map(lambda raw: float(raw.replace(',','.')))
    sql_data['TS_or_Distance'] = sql_data['TS_or_Distance'].astype(float)
    sql_data.sort_values(by ='TS_or_Distance', inplace=True)
    sql_data.reset_index(drop = True, inplace=True)
    
    if is_ARAN:
        drop_cols = ['DRDMeasurementId', 'T', 'isComputed', 'FK_Trip', 'FK_MeasurementType', 'Created_Date',
       'Updated_Date','BeginChainage','EndChainage']
        extract_string_column(sql_data)
        sql_data.drop(drop_cols, axis=1, inplace = True)
         
    if is_p79:
        iri =  sql_data[sql_data['T']=='IRI']
        iri['IRI_mean'] = iri.message.apply(lambda message: (loads(message)['IRI5']+loads(message)['IRI21'])/2)
        iri.drop(columns=['message','DRDMeasurementId', 'T',],inplace=True,axis=1)
        
        # Filter iri
        iri = iri[(iri.lat>0) & (iri.lon>0)]
        iri.reset_index(drop=True, inplace=True)
    
    # Get information about the trip
    print('Getting trip information')
    quory = 'SELECT * FROM "Trips"'
    #quory = 'SELECT * FROM "Trips" WHERE "TaskId"=\'0\''
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn) 
    
    # Close connection
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
        
    # Return
    if is_p79:   
        return sql_data, iri, trips
    elif is_ARAN:
        return sql_data, None, trips
        

def filter_latlon(data, col_string, lat_min, lat_max, lon_min, lon_max):
    data = data[data['lat'].between(lat_min,lat_max)]
    data = data[data['lon'].between(lon_min,lon_max)]
    data.reset_index(inplace=True, drop=True)
    return data


def select_platoon(filename):
    id = filename.split('/')[1].split('GMtrip-')[1].split('_DRD')[0]
    platoon_ids = ['4955','4957','4959']
    if id in platoon_ids:
        return True
    else:
        return False
    
def get_filenames(input_dir, mode, filetype='pkl'):
    # mode: accspeed_all
    
    all_filenames = glob.glob('{0}/*.{1}'.format(input_dir, filetype))
    print(all_filenames)
    
    if mode =='acc': #all files
        return all_filenames
    elif (mode=='speed' or mode=='accspeed'):
        return [filename for filename in all_filenames if 'accspeed' in filename]
    elif mode=='platoon_all':
        return list(filter(select_platoon, all_filenames))
    


def prepare_data(data, target_name = 'DRD_IRI_mean', bins = [0,2,5,50]):
    data['DRD_IRI_mean'] = data.apply(lambda x: (x['DRD_IRI5']+x['DRD_IRI21'])/2, axis=1)
    data['len']=data.GM_Time_segment.apply(lambda row:row.shape[0])
    data['GM_time_start'] = data.GM_Time_segment.apply(lambda t:t[0]) 
    data['GM_time_end'] = data.GM_Time_segment.apply(lambda t:t[-1]) 
    
    # Filter and classify
    data = data[data['len']<5000]
    data = set_class(data, target_name = target_name, bins = bins)
    data.reset_index(drop=True, inplace=True)
    
    return data

def get_segments(data, window_size = 10, step = 10, is_aran = False, is_aran_sel = False):
    
    # Take start and end or mean for those columns
    if is_aran:
        cols_to_take_ends = ['TS_or_Distance', 'lat', 'lon', 'lat_map', 'lon_map','street_name']
        ignore_cols = ['BeginDistanceStamp','EndDistanceStamp']
        cols_to_average = []
        other_cols = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols+cols_to_average)) )
        ts_column = 'TS_or_Distance'
    elif is_aran_sel:
        cols_to_take_ends = ['TS_or_Distance_start', 'TS_or_Distance_end', 'lat_start', 'lon_start',
       'lat_end', 'lon_end', 'lat_map_start', 'lon_map_start', 'lat_map_end',
       'lon_map_end', 'street_name_start', 'street_name_end']
        ignore_cols = []
        other_cols = []
        cols_to_average = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols+ ignore_cols+other_cols)) )
        ts_column = 'TS_or_Distance_start' 
    else:
        cols_to_take_ends = []
        ignore_cols = []
        cols_to_average = []
        cols_to_average = []
        other_cols = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols+cols_to_average)) )
    
    # Final df
    final_cols = [col + '_start' for col in cols_to_take_ends] + [col + '_end' for col in cols_to_take_ends] + other_cols
    s = pd.DataFrame(columns=final_cols)
    
    # First and last TS_or_Distance divisable by window_size
    i=0
    first = data[ts_column].iloc[0]
    while first%step!=0:
        first = data[ ts_column].iloc[i]
        i = i+1
    
    i=-1
    last = data[ts_column].iloc[-1]
    while last%window_size!=0:
        last = data[ ts_column].iloc[i]
        i = i-1
        
        
    # Make segments
    segments = []
    row=0
    start_ts = first
    while start_ts < last:
        end_ts = start_ts + window_size
            
        # Print
        #print(start_ts, end_ts)
        
        #if start_ts%500==0:
        #    print(start_ts, end_ts)
        if is_aran_sel==True:
            c1 = data[ts_column] >= start_ts
            c2 = data[ts_column.replace('_start','_end')] <= end_ts
            data_this_seg = data[c1 & c2]
        else:
            data_this_seg = data[data[ts_column].between(start_ts, end_ts)]
        
        # Start, end
        for col in cols_to_take_ends:  
            s.at[row, col+'_start'] = data_this_seg[col].iloc[0]
            s.at[row, col+'_end'] = data_this_seg[col].iloc[-1]
        
        # Means
        for col in cols_to_average:
            #print('mean for: ',col,data_this_seg[col].mean())
            s.at[row,col] = data_this_seg[col].mean()
            
             
        # Sums
        for col in other_cols:
            s.at[row,col] = data_this_seg[col].sum()
        
        row = row+1
        start_ts = start_ts + step

    return s

  
