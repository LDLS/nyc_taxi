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