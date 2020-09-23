import os
import urllib
import zipfile
import itertools
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import shapefile
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
style.use('ggplot')

PARENT_DIR = Path().resolve().parents[0] 
DATA_DIR = PARENT_DIR / 'data'
DATA_URL = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'

data_name = 'tripdata'
companies = ['yellow']
years = ['2017']
months = ['03', '06', '11']
cos = ''.join([c[0] for c in set(companies)])
yrs = ''.join([y[-2:] for y in set(years)])
mos = ''.join(months)
dates = ['-'.join([y,m]) for y, m in itertools.product(years, months)]

taxi_data_csv_names = ['_'.join([c, data_name, d]) + '.csv'
                       for c, d in itertools.product(companies, dates)]
taxi_data_pattern = DATA_DIR / '*_tripdata_*-*.csv'
csv_urls = [DATA_URL + f for f in taxi_data_csv_names]
csv_files = [DATA_DIR / f for f in taxi_data_csv_names]
csv_dict = dict(zip(taxi_data_csv_names, zip(csv_urls, csv_files)))
parquet_name = DATA_DIR / '_'.join([cos, yrs, mos])

lookup_name = DATA_DIR / 'taxi_zone_lookup.csv'
shapefile_name = DATA_DIR / 'taxi_zones.zip'

numerical_cols = ['passenger_count', 'trip_distance', 'fare_amount',
                  'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                  'improvement_surcharge', 'total_amount']

categorical_cols = ['VendorID', 'RatecodeID',
                    'store_and_fwd_flag', 'payment_type']

location_cols = ['PULocationID', 'DOLocationID']

datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']

categorical_dict = {'VendorID': {1: 'Creative Mobile',
                                 2: 'VeriFone'},
                    'RatecodeID': {1: 'Standard',
                                   2: 'JFK',
                                   3: 'Newark',
                                   4: 'Nassau/Westchester',
                                   5: 'Negotiated',
                                   6: 'Group'},
                    'store_and_fwd_flag': {0: 'not store and forward',
                                           1: 'store and forward'},
                    'payment_type': {1: 'Credit',
                                     2: 'Cash',
                                     3: 'No charge',
                                     4: 'Dispute',
                                     5: 'Unknown',
                                     6: 'Voided'}}


def dwnld_data(csv_name):
    ''' Downloads file from the url and save it as filename '''
    # check if file already exists
    url, filename = csv_dict[csv_name]
    if not filename.exists():
        print(f'Downloading {csv_name}')
        urllib.request.urlretrieve(url, filename)
    else:
        print(f'{csv_name} exists')


def dwnl_location_lookup():
    url = 'https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv'
    filename = lookup_name
    if not filename.exists():
        print('Downloading lookup table')
        urllib.request.urlretrieve(url, filename)
    else:
        print('Lookup table exists')


def dwnl_extract_shapefiles():
    url = 'https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip'
    filename = shapefile_name
    base_dir = DATA_DIR
    if not filename.exists():
        print('Downloading shapefiles')
        urllib.request.urlretrieve(url, filename)
    else:
        print('Shapefile exists')
    filezip = zipfile.ZipFile(filename, 'r')
    filezip_unzip_fn = [(DATA_DIR / f).exists()
                        for f in filezip.namelist()]
    all_contents = all(filezip_unzip_fn)
    if not all_contents:
        print('Extracting shapefile')
        filezip.extractall(base_dir)
    else:
         print('Contents exist')


def write_parquet():
    df = dd.read_csv(taxi_data_pattern, delimiter=',')
    df_initial_n_rows = df.shape[0]

    # Drop rows with all entries equal to NAN
    df = df.dropna('all')

    df.store_and_fwd_flag = df.store_and_fwd_flag.replace({'Y': 1, 'N': 0})
    df[categorical_cols] = df[categorical_cols].astype('uint8')
    df[location_cols] = df[location_cols].astype('uint16')
    df[datetime_cols] = df[datetime_cols].astype('M8[us]')
    df[numerical_cols] = df[numerical_cols].astype('float32')
    df.passenger_count = df.passenger_count.astype('uint8')
    
    if not parquet_name.exists():
        print('Saving parquet file')
        with ProgressBar():
            df.to_parquet(parquet_name, compression='snappy')
    else:
        print('Parquet file already exists')