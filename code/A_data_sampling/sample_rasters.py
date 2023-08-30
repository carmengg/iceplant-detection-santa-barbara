import os
import pandas as pd
import numpy as np

import pystac_client 
import planetary_computer as pc

import calendar

from shapely.geometry import shape
from shapely.geometry import Point

import random
#random.seed(10)

import warnings

import rioxarray as rioxr
import geopandas as gpd

import rasterio
from rasterio.crs import CRS

from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import minimum_filter as minf2D
from scipy.ndimage import convolve as conf2D

from skimage.morphology import disk
from skimage.filters.rank import entropy


# *********************************************************************
def get_item_from_id(itemid):
    """
        Searches the Planetary Computer's NAIP collection for the item associated with the given itemid.
            Parameters:
                        itemid (str): the itemid of a single NAIP scene
            Returns:
                        item (pystac.item.Item): item associated to given itemid (unsigned)
   """
    # accesing Planetary Computer's storage using pystac client
    URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(URL)

    search = catalog.search(collections=["naip"], ids = itemid)
    
    # return 1st item in search (assumes itemid IS associaed to some item)
    return list(search.items())[0]   # ** TO DO: catch exception

# ---------------------------------------------

def get_raster_from_item(item):
    """
        "Opens" the raster in the given item: returns a rasterio.io.DatasetReader to the raster in item.
            Parameters: item (pystac.item.Item)
            Returns: reader (rasterio.io.DatasetReader) 
    """  
    href = pc.sign(item.assets["image"].href)
    reader = rasterio.open(href)
    return reader

# ---------------------------------------------

# TO DO: not used
def get_crs_from_itemid(itemid):
    # accesing Azure storage using pystac client
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # search for naip scene where the pts were sampled from
    search = catalog.search(
        collections=["naip"],
        ids = itemid
    )
    item = list(search.items())[0]
    epsg_code = item.properties['proj:epsg']
    return  CRS.from_epsg(epsg_code)

# *********************************************************************

def day_in_year(day,month,year):
    """
        Transforms a date into a day in the year from 1 to 365/366 (takes into account leap years).
            Paratmeters:
                day (int): day of the date
                month (int 1-12): month of the date
                year (int): year of date
            Returns:
                n (int): date as day in year
    """
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    n = 0
    for i in range(month-1):
        n = n+days_in_month[i]
    n = n+day
    if calendar.isleap(year) and month>2:
        n = n+1
    return n


# *********************************************************************

def count_pixels_in_polygons(polys, rast_reader):
    """
        Counts the approximate number of pixels in a raster covered by each polygon in a list.
        No need to match CRS: to do the count it internally matches the CRS of the polygons and the raster. 
            Parameters:
                        polys (geopandas.geodataframe.GeoDataFrame): 
                            GeoDataFrame with geometry column of type shapely.geometry.polygon.Polygon
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster on which we will "overlay" the polygons to count the pixels covered
            Returns:
                    n_pixels (numpy.ndarray): 
                        approximate number of pixels from raster covered by each polygon
            
    """
    # convert to same crs as raster to properly calculate area of polygons
    if polys.crs != rast_reader.crs:
        print('matched crs')
        polys = polys.to_crs(rast_reader.crs)
    
    # area of a single pixel from raster resolution    
    pixel_size = rast_reader.res[0]*rast_reader.res[1]
    
    # get approx number of pixels by dividing polygon area by pixel_size
    n_pixels = polys.geometry.apply(lambda p: int((p.area/pixel_size)))
    
    return  n_pixels.to_numpy()


# *********************************************************************

# N = number of polygons
# S = total number of pts to sample
def staggered_n_samples(N,S):
    if N>S:
        X = np.zeros(N)
        X[:S]+=1
        return X
    
    n = 0     # width of steps
    P_n = S+1 # number of pts needed to assemble the "ladder" with smallest steps
    while P_n>S:
        n +=1
        q = int(N/n)
        r = int(N%n)
        P_n = n*(q*(q+1)/2) + r
        
    X = []
    for i in range(q,0,-1):
        X.append(np.full(n,i))
    X.append(np.full(r,1))
    X = np.concatenate(X)
    
    # distribute remaining points by filling each level from biggest poly to smallest
    R = S - P_n
    qR = int(R/N)
    rR = int(R%N)
    
    X +=qR
    X[:rR]+=1
    return X


# *********************************************************************

def sample_size_in_polygons(n_pixels, total_pts):
    """
        Calculates the number of points to sample from each polygon in a list 
        by distributing total_pts proportionally across polygons according to polygon area.
            
            Parameters:
                       n_pixels (numpy.ndarray):
                           array with (approximate) number of pixels contained in each polygon 
                        total_pts (int):
                            how many points to sample, in total, from polygons
            Returns:
                    n_pts (numpy.ndarray): 
                        array with number of pts to sample from each polygon
    """
    
    X = staggered_n_samples(len(n_pixels), total_pts)
    X.sort()
    df = pd.DataFrame({'n_pixels' : n_pixels}).sort_values(by = ['n_pixels'])
    df['n_sample']= X
    df = df.sort_index()
    n_pts = df.n_sample.to_numpy()
    n_pts = n_pts.astype('int')
    
    return n_pts

# *********************************************************************

def random_pts_poly(N, polygon):
    """
        Creates a list of N points sampled randomly from within the given polygon.
            Parameters:
                        N (int): number of random points to sample form polygon
                        polygon (shapely.geometry.polygon.Polygon): polygon from which to sample points
            Return:
                    points (list of shapely.geometry.point.Point): 
                        list of N random points sampled from polygon
    """
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < N:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points  

# *********************************************************************

def sample_raster_from_poly(N, poly, poly_id, class_name, poly_class, rast_reader, rast_band_names, rast_crs):
    """
        Creates a dataframe of raster bands values at N points randomly sampled from a given polygon.
        Polygon and raster must have SAME CRS for results to be correct. 
        Resulting dataframe includes metadata about the sampled polygon: poly_id, poly_class.
        
            Parameters:
                        N (int): 
                            number of random points to sample form polygon
                        poly (shapely.geometry.polygon.Polygon): 
                            polygon from which to sample points
                        poly_id (int): 
                            id number of the polygon
                        class_name (str): 
                            name of the classification in which polygons outline pixels from same class(ex: 'land_cover')
                        poly_class (int): 
                            class of the data represented by polygon (ex: 1 if polygon is building, 2 if it's water, etc)
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to extract band information at every point sampled from polygon
                        rast_band_names (str list):
                            names of the bands of rast_reader
                        rast_crs (str):
                            CRS of rast_reader
            Returns: 
                    sample (pandas.core.frame.DataFrame): data frame of raster bands' values at the N points sampled from poly.

    """
    # TO DO: add catch when polygon and raster do not intersect
    
    # select random points inside poly
    points = random_pts_poly(N, poly)  
    
    # make data frame with sampled points
    sample = pd.DataFrame({           
        'geometry': pd.Series(points), 
        class_name : pd.Series(np.full(N, poly_class)), 
        'polygon_id': pd.Series(np.full(N, poly_id))
                 })
    # separate coords (needed for rasterio.io.DatasetReader.sample() )
    sample_coords = sample.geometry.apply(lambda p: (p.x, p.y))  
    data_generator = rast_reader.sample(sample_coords)

    # make band values into dataframe
    data = np.vstack(list(data_generator))               
    data = pd.DataFrame(data, columns=rast_band_names) 
    
    # add band data to sampled points
    sample = pd.concat([sample,data],axis=1)  
    
    kwargs = {'x' : sample.geometry.apply(lambda p : p.x),
             'y' : sample.geometry.apply(lambda p : p.y),
             'pts_crs' : rast_crs}
    sample = sample.assign(**kwargs)    
    sample = sample.drop('geometry', axis=1)

    # organize columns
    sample = sample[['x','y','pts_crs','polygon_id', class_name] + rast_band_names] 

    return sample

# *********************************************************************
def sample_naip_from_polys(polys, class_name, itemid, total_pts):
    """
        Creates a dataframe of given NAIP scene's bands values at points sampled randomly from polygons in given list.
        Resulting dataframe includes metadata about the sampled polygons and NAIP raster.
        No need to match polygons and raster CRS, this is done internally.
        
        Parameters:
                        polys (geopandas.geodataframe.GeoDataFrame): 
                            GeoDataFrame with geometry column of type shapely.geometry.polygon.Polygon
                            Index must begin at 0.
                        class_name (str): 
                            name of column in polys GeoDataFrame having the classification in which polygons outline pixels from same class (ex: 'land_cover')
                        itemid (str): 
                            the itemid of a single NAIP scene over which the polygons with be "overlayed" to do the data sampling
                                                total_pts (int):
                            how many points to sample, in total, from polygons
            Return:
                    df (pandas.core.frame.DataFrame): data frame of raster bands' values at points sampled from polys.

    """    
    item = get_item_from_id(itemid)
    
    rast_reader = get_raster_from_item(item)        
    rast_band_names = ['r','g','b','nir']
    rast_crs = rast_reader.crs.to_dict()['init']
    
    polys_match = polys.to_crs(rast_reader.crs)
    
    n_pixels = count_pixels_in_polygons(polys_match, rast_reader)
    n_pts = sample_size_in_polygons(n_pixels, total_pts)
    
    samples = []
    for i in range(polys.shape[0]):
        if n_pts[i]>0:
            sample = sample_raster_from_poly(n_pts[i], 
                                             polys_match.geometry[i], 
                                             polys.id[i], 
                                             class_name, 
                                             polys[class_name][i], 
                                             rast_reader, 
                                             rast_band_names, 
                                             rast_crs)                                   
            samples.append(sample)   
    # create dataframe from samples list        
    df = pd.concat(samples) 
    
    kwargs = {'year' : item.datetime.year,
             'month' : item.datetime.month,
             'day_in_year' : day_in_year(item.datetime.day, item.datetime.month, item.datetime.year),
             'naip_id' : itemid}
    df = df.assign(**kwargs)     
    df = df.reset_index(drop=True)
    
    return df


# *********************************************************************


def sample_naip_from_polys_no_warnings(polys, class_name, itemid, total_pts):
    """
       Runs sample_naip_from_polys function catching the following warning:
                   /srv/conda/envs/notebook/lib/python3.8/site-packages/pandas/core/dtypes/cast.py:122: ShapelyDeprecationWarning: 
                   The array interface is deprecated and will no longer work in Shapely 2.0. 
                   Convert the '.coords' to a numpy array instead. arr = construct_1d_object_array_from_listlike(values)
        # See https://shapely.readthedocs.io/en/stable/migration.html, section Creating NumPy arrays of geometry objects

            Parameters: see parameters for sample_naip_from_polys function
            Return: see return for sample_naip_from_polys function
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sample_naip_from_polys(polys, class_name, itemid, total_pts)
    return df


# *********************************************************************
def geodataframe_from_csv(df=None, fp=None, lon_label=None, lat_label=None, crs=None):
    """
        Transforms a csv with longitude and latitude columns into a GeoDataFrame.
            Parameters:
                        fp (str): 
                            File path to csv containing coordinates of points.    # TO DO: update to pandas data frame, don't read in file here
                            The coordiantes must be in separate columns. 
                        lon_label (str): 
                            name of column having longitudes of points
                        lat_label (str): 
                            name of column having latitudes of points
                        crs (rasterio.crs.CRS):
                            crs of points in csv. All points must be in the same crs.
            Returns:
                     geopandas.geodataframe.GeoDataFrame:
                        the csv in given file path converted to a GeoDataFrame with geometry column 
                        of type shapely.geometry.Point constructed from longitude and latitude columns
    """
    if df is None:
        if fp is not None:
            df = pd.read_csv(fp)
        else:
            return False
    # rename geometry column if it exists        
    if 'geometry' in df.columns:          
        df = df.rename(columns = {'geometry': 'geometry_0'})

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_label],df[lat_label]), crs=crs)

# *********************************************************************

def sample_raster_from_pts(pts, rast_reader, rast_band_names):
    """
        Creates a dataframe of raster bands values at the given points.
            Parameters: 
                pts (geopandas.geoseries.GeoSeries): 
                    GeoSeries of the points  (type shapely.geometry.point.Point) where the samples from the rasters will be taken
                rast_reader (rasterio.io.DatasetReader):
                    reader to the raster from which to sample bands
                rast_band_names (str list):
                    names of the bands of rast_reader
            Return:
                samples (pandas.core.frame.DataFrame): data frame of raster bands' values at the given points

    """
    if rast_reader.count != len(rast_band_names):
        print('# band names != # bands in raster')
        return
    
    pts_match = pts.to_crs(rast_reader.crs)

    # sample
    sample_coords = pts_match.apply(lambda p :(p.x, p.y))  
    samples_generator = rast_reader.sample(sample_coords)    
    
    # make band values into dataframe
    samples = np.vstack(list(samples_generator))   
    samples = pd.DataFrame(samples, columns=rast_band_names)
    
    return samples

# *********************************************************************

def open_and_match(fp, reproject_to):
    rast = rioxr.open_rasterio(fp)
    rast_match = rast.rio.reproject_match(reproject_to)
    return rast_match.squeeze()

# ------------------------------------------------------------------------------

def input_raster(rast_reader=None, raster=None, band=1, rast_data=None, crs=None, transf=None):
    
    if rast_reader is not None:
        rast = rast_reader.read([band]).squeeze() # read raster values
        crs = rast_reader.crs
        transf = rast_reader.transform
        
    elif raster is not None:
        if len(raster.shape) == 3: # multiband raster            
            rast = raster[band-1].squeeze()
        elif len(raster.shape) == 2: #one band raster
            rast = raster
        crs = raster.rio.crs
        transf = raster.rio.transform()
    
    elif rast_data is not None:
        rast = rast_data
        crs = crs
        transf = transf
    else:
        return 
    
    return rast, crs, transf

# *********************************************************************
def make_directory(dir_name): 
    """ 
        Checks if the directory with name dir_name (str) exists in the current working directory. 
        If it doesn't, it creates the directory and returns the filepath to it.
    """    
    fp = os.path.join(os.getcwd(),dir_name)  
    if not os.path.exists(fp):
        os.mkdir(fp)
    return fp

# ------------------------------------------------------------------------------
def save_raster(raster, fp, shape, bands_n, crs, transform, dtype):
    """
        Saves an array as a 'GTiff' raster with specified parameters.
        Parameters:
                    raster (numpy.ndarray): array of raster values
                    fp (str): file path where raster will be saved, including file name
                    shape (tuple):shape of raster (height, width) TO DO: SHOULD THIS BE READ DIRECTLY FROM raster??
                    bands_n (integer): number of bands in the raster
                    crs (str): CRS of raster
                    transform (affine.Affine): affine transformation of raster
        Return: None
    """
    bands_array = 1
    if bands_n > 1:
        bands_array = np.arange(1,bands_n+1)
        
    with rasterio.open(
        fp,  # file path
        'w',           # w = write
        driver = 'GTiff', # format
        height = shape[0], 
        width = shape[1],
        count = bands_n,  # number of raster bands in the dataset
        dtype = dtype,
        crs = crs,
        transform = transform,
    ) as dst:
        dst.write(raster.astype(dtype), bands_array)
    return 

# ------------------------------------------------------------------------------
def save_raster_checkpoints(rast, crs, transf, rast_name=None, suffix= None, folder_path=None):  

    if rast_name is None:
        rast_name = 'raster'        

    if (folder_path is None) or (os.path.exists(folder_path) == False):  
        folder_path = make_directory('temp')  

    if suffix is None:
        suffix = ''
    else:
        suffix = '_'+suffix
        
    fp = os.path.join(folder_path, rast_name + suffix + '.tif')      

    dtype = rasterio.dtypes.get_minimum_dtype(rast)      

    save_raster(rast, 
                fp, 
                rast.shape,
                1,
                crs, 
                transf, 
                dtype) 
    return 

# *********************************************************************

def min_raster(rast_reader=None, raster=None,  rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):  
    """
        Creates a new raster by replacing each pixel p in given raster R by the minimum value in a nxn window centered at p.
        The raster with minimum values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the minimum values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_maxs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute minimum values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)

    if rasterio.dtypes.get_minimum_dtype(rast)  == 'uint8':
        cval = 255
    else:
        cval = 0
        
    mins = minf2D(rast, size=(n,n), cval=cval)     
    
    save_raster_checkpoints(mins, crs, transf, rast_name, 'mins', folder_path)
    return

# ------------------------------------------------------------------------------

def max_raster(rast_reader=None, raster=None,  rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):  
    """
        Creates a new raster by replacing each pixel p in given raster R by the max value in a nxn window centered at p.
        The raster with maximum values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the maximum values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_maxs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute maximum values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)
        
    maxs = maxf2D(rast, size=(n,n))
    
    save_raster_checkpoints(maxs, crs, transf, rast_name, 'maxs', folder_path)
    return

# ------------------------------------------------------------------------------

def avg_raster(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None): 
    """
        Creates a new raster by replacing each pixel p in given raster R by the avg value in a nxn window centered at p.
        The raster with averege values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the average values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_avgs.tif.
                        n (int):
                            Side length (in pixels) of the square window over which to compute average values for each pixel.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)

    w = np.ones(n*n).reshape(n,n)      
    avgs = conf2D(rast, 
             weights=w,
             mode='constant',
             output='int64')
    avgs = avgs/(n**2)

    save_raster_checkpoints(avgs, crs, transf, rast_name, 'avgs', folder_path)            
    return
                      
# ------------------------------------------------------------------------------

def entropy_raster(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=2, folder_path=None): 
    """
        Creates a new raster by replacing each pixel p in given raster R by the entropy value in a disk of radius n centered at p.
        The raster with entropies values is saved in a temp folder in the current working directory if no folder_path is given.
            Parameters: 
                        rast_reader (rasterio.io.DatasetReader):
                            reader to the raster from which to compute the average values in a window
                        rast_name (str):
                            name of raster. The resulting raster will be saved as rast_name_avgs.tif.
                        n (int):
                            radius of disk over which to calculate entropy.
                        folder_path (str):
                            directory where to save raster. If none is given, then it saves the raster in a temp folder in the cwd.
            Return: None    
    """
    
    rast, crs, transf = input_raster(rast_reader, raster, band, rast_data, crs, transf)
    
    entropies = entropy(rast, disk(n))    
    
    save_raster_checkpoints(entropies, crs, transf, rast_name, 'entrs', folder_path)            
    return
# ------------------------------------------------------------------------------

def max_min_avg_rasters(rast_reader=None, raster=None, rast_data=None, crs=None, transf=None, band=1, rast_name=None, n=3, folder_path=None):
    max_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    min_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    avg_raster(rast_reader, raster, rast_data, crs, transf, band, rast_name, n, folder_path)
    return

# *********************************************************************
# *********************************************************************
# *********************************************************************

# Functions from previous raster_to_features

# --- print proportions of ice plant (1) vs no iceplant (0) in an array with only 0 and 1
# TO DO: change to label_proportions
def iceplant_proportions(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print('no-iceplant:iceplant ratio    ',round(counts[0]/counts[1],1),':1')
    n = labels.shape[0]
    perc = [round(counts[0]/n*100,2), round(counts[1]/n*100,2)]
    df = pd.DataFrame({'iceplant':unique,
             'counts':counts,
             'percentage':perc}).set_index('iceplant')
    print(df)
    print()
    

# ---------------------------------------------
# rast is 4 band xarray
def ndvi_xarray(rast):
    red_band = rast.sel(band=1).astype('int16') 
    nir_band = rast.sel(band=4).astype('int16')
    return (nir_band - red_band) / (nir_band + red_band)

# **********************************************************************************************************
# **********************************************************************************************************

def rioxr_from_itemid(itemid, reduce_box = None, reduce_box_crs = None):
    """
        Opens the raster associated with the given itemid. 
        If a reduce_box is given, then it opens the subset of raster determined by teh box.
            Parameters: 
                        itemid (str): the itemid of a scene in the planetary computer data repository
                        reduce_box (shapely.geometry.polygon.Polygon): 
                            box outlining the perimeter of the area of interest within the scene
                        reduce_box_crs (str):
                            CRS of reduce_box
            Return: 
                    xarray.core.dataarray.DataArray : rioxarray of scene or a subset of it.
    """
    item = get_item_from_id(itemid)    # locate raster
    href = pc.sign(item.assets["image"].href)
    
    rast = rioxr.open_rasterio(href)           # open raster
    
    if reduce_box != None:
        reduce = gpd.GeoDataFrame({'geometry':[reduce_box]}, crs=reduce_box_crs)    # clip if needed
        reduce = reduce.to_crs(rast.rio.crs)        
        rast = rast.rio.clip_box(*reduce.total_bounds)
    
    rast.attrs['datetime'] = item.datetime    # add date of collection
    
    return rast

# **********************************************************************************************************

def raster_as_df(raster, band_names):
    """
        Transforms the given raster into a dataframe of the pixels with column names equal to the band ndames.
             Parameters:
                       raster (numpy.ndarray): raster values
                       band_names (list): names (str) of band names. order of names must be the same order as bands.
            Returns: 
                    df (pandas.core.frame.DataFrame): dataframe where each pixels is a row and columns are the 
                    rasters's band values at pixel
    """  
    pixels = raster.reshape([len(band_names),-1]).T
    df = pd.DataFrame(pixels, columns=band_names) 
    return df

# **********************************************************************************************************
def normalized_difference_index(df, *args):
    """f
        Calculates the normalized difference index of two columns in the given data frame.
        In doing so it converts the column types to int16 (spectral bands are usually uint8).
            Parameters:
                        df (pandas.core.frame.DataFrame): dataframe from which two columns will be used
                            to calculate a normalized difference index
                        *args: tuple of column indices used as x and y in normalized difference
            Returns:
                    pandas.core.series.Series: the normalized difference index of the selected columns
                    
            Example: for dataframe with columns red, green, blue, nir (in that order)
                     ndvi would be normalized_difference_index(df, 3,0), and
                     ndwi would be normalized_difference_index(df, 1,3)
    """    
    m = args[0]
    n = args[1]
    
    x = df.iloc[:, m].astype('int16')  
    y = df.iloc[:, n].astype('int16')
    return (x-y) / (x+y)

# **********************************************************************************************************

def feature_df_treshold(df, feature_name, thresh, keep_gr, func, *args):
    """
        Adds a new column C to a dataframe using the action of a function and 
        selects only the rows that whose values in C are above a certain threshold.
            Parameters: 
                        df (pandas.core.frame.DataFrame): data frame on which to do the operation
                        feature_name (str): name of new column
                        thresh (float): threshold for new column
                        keep_gr (bool): if keep_gr == True then it keeps the rows with new_column > thresh
                                        if keep_gr == False then it keeps the rows with new_column < thresh
                        func (function): function to calculate new column in dataframe
                        *args: arguments for function 
            Returns:
                    keep (pandas.core.frame.DataFrame):
                        a copy of dataframe with the values of function as a new column and subsetted by threshold
                    deleted_indices (numpy.ndarray): 
                        indices of the rows that were deleted from df 
                        (those with value of function not compatible with threshold condition)s
                        
    """
    # add new column
    kwargs = {feature_name : func(df, *args)}        # TO DO: maybe take these two lines out?
    df = df.assign(**kwargs)
    
    # select rows above threshold, keep indices of deleted rows
    if keep_gr == True:
        keep = df[df[feature_name] > thresh]
        deleted_indices = df[df[feature_name] <= thresh].index
    # select rows below threshold, keep indices of deleted rows
    else : 
        keep = df[df[feature_name] < thresh]
        deleted_indices = df[df[feature_name] >= thresh].index
        
    deleted_indices = deleted_indices.to_numpy()
    
    return keep, deleted_indices

# **********************************************************************************************************

def add_spectral_features(df, ndwi_thresh, ndvi_thresh):
    """
       Finds the rows in df with ndwi values below ndwi_thresh and ndvi values above ndvi_thresh. 
       Keeps track of the rows deleted.
           Parameters:
                       df (pandas.core.frame.DataFrame): dataframe with columns red, green, blue and nir (in that order)
                       ndwi_tresh (float): threshold for ndwi
                       ndvi_tresh (float): threshold for ndvi
           Returns: 
                    is_veg (pandas.core.frame.DataFrame):
                        subset of df in which all rows have ndwi values below ndwi_thresh and ndvi values above ndvi_thresh
                    water_index (numpy.ndarray): 
                        indices of rows in df with ndwi values above ndwi_thresh 
                    not_veg_index (numpy.ndarray): 
                        indices of rows in df with ndwi values below ndwi_thresh and ndvi values below ndvi_thresh
    """
    # remove water pixels
    not_water, water_index = feature_df_treshold(df, 
                                             'ndwi', ndwi_thresh, False, 
                                             normalized_difference_index, 1,3)   
    # remove non-vegetation pixels
    is_veg, not_veg_index = feature_df_treshold(not_water, 
                                                   'ndvi', ndvi_thresh, True, 
                                                   normalized_difference_index, 3,0)
    # return pixels that are vegetation and are not water
    # return indices of water pixels and not vegetation pixels
    return is_veg, water_index, not_veg_index

# ----------------------------------------------------

def add_date_features(df, date): 
    """
        Adds three constant columns to the data frame df with info from date (datetime.datetime): year, month and day_in_year.
    """
    kwargs = {'year' : date.year,
             'month' : date.month,
             'day_in_year' : day_in_year(date.day, date.month, date.year)}
    
    return df.assign(**kwargs)

# **********************************************************************************************************

def indices_to_image(nrows, ncols, indices_list, values, back_value):
    """
        Parameters:
                    nrows (int): number of rows in ouput array
                    ncols (int): number of columns in output array
                    indices_list (list): 
                        list of 1-dimensional np.arrays. 
                        each element in list must be values within [0, nrows*ncols-1], 
                        these represent cells in the output array with same value
                    values (list): the value to assign to each of the arrays in indices_list
                    back_value (int): value of any cell not in the union of indices_list
            Returns:
                reconstruct (numpy.ndarray):
                    array with values in valuesU{back_value} with dimensions nrows*ncols 
                    the cells with 'index' in the array indices_list[i] get assigned the value values[i]
                    the index of a i,j cell of the array is i*nrows+j
            Example:
                Suppose nrows=ncols=3, indices_list = [[2,3,7],[0,1,8]], values = [1,2] and back_value =0. 
                Then the output is the array 
                |2|2|1|
                |1|0|0|
                |0|1|2|
                        
    """
    # background, any pixel not in the union of indices will be given this value
    reconstruct = np.ones((nrows,ncols))*back_value 

    # TO DO: check indices list and values lengths are the same?
    for k in range(0,len(indices_list)):
        i = indices_list[k] / ncols
        i = i.astype(int)
        j = indices_list[k] % ncols
        reconstruct[i,j] = values[k]
    
    return reconstruct

# **********************************************************************************************************

def create_aux_canopyheight_rasters(year):
    # open canopy height raster for given year
    lidar_rast_reader = rasterio.open(path_to_lidar(year))   
    # name of output canopy height raster
    rast_name = 'SB_canopy_height_' + str(year) 

    # if there is no temp folder, create one
    temp_fp = os.path.join(os.getcwd(), 'temp') 
    if not os.path.exists(temp_fp):
        os.mkdir(temp_fp)
    temp_fp = os.path.join(temp_fp, 'aux_canopy_height')
    if not os.path.exists(temp_fp):
        os.mkdir(temp_fp)

    # list of file paths to aux canopy height rasters
    # order of filepaths is: lidar, max, min, avg
    lidar_fps = [path_to_lidar(year)]  
    for tag in ['_maxs', '_mins', '_avgs']:
        lidar_fps.append(os.path.join(temp_fp, rast_name + tag + '.tif'))

    # create any missing aux canopy height rasters
    if not all([os.path.exists(fp) for fp in lidar_fps]):
        # save aux rasters in temp folder
        if not os.path.exists(lidar_fps[1]):  # starts at 1 bc 0 is canopy height raster
            max_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)

        if not os.path.exists(lidar_fps[2]):
            min_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)  

        if not os.path.exists(lidar_fps[3]):
            avg_raster(rast_reader = lidar_rast_reader, rast_name = rast_name, n=3, folder_path=temp_fp)
    return lidar_fps

# **********************************************************************************************************
def finish_processing(status, processed, reason, times_features, times_class, times_post, veg_pixels, itemid):
    
    processed.append('N')
    times_features.append(0)
    times_class.append(0)        
    times_post.append(0)
    veg_pixels.append(0)
    
    if status == 'no_data':
        reason.append('no data in intersection')
    elif status == 'no_veg':
        reason.append('no vegeatation in intersection')
    else: 
        reason.append('invalid status') 
    
    return
    
def finish_processing_message(status, itemid):
    if status == 'no_data':
        print('no data at intersection of scene with coastal buffer')  
    elif status == 'no_veg':
        print('no vegetation pixels at intersection of scene data with coastal buffer')
    else: 
        print('invalid status')
        return 
    
    print('FINISHED: ', itemid , '\n', end="\r")
    return