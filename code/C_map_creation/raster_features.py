import os
import pandas as pd
import numpy as np

import pystac_client 
import planetary_computer as pc

import calendar

import rioxarray as rioxr
import geopandas as gpd

import rasterio
from rasterio.crs import CRS

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

# *********************************************************************
def ndvi_xarray(rast):
    """Calculates the 'NDVI' in a raster rast with 4 bands"""
    red_band = rast.sel(band=1).astype('int16') 
    nir_band = rast.sel(band=4).astype('int16')
    return (nir_band - red_band) / (nir_band + red_band)

# *********************************************************************

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

# *********************************************************************

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

# *********************************************************************
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

# *********************************************************************

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

# *********************************************************************

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

# *********************************************************************

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

# *********************************************************************
def finish_processing(status, processed, reason, times_features, times_class, times_post, veg_pixels, itemid):
    """An auxiliary function to print messages and add 0 computation times 
       when there's nothing to classify in a scene 
    """
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