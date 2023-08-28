import os
import requests
import geemap
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from natsort import natsorted
from pyproj import CRS
import shutil
import warnings
warnings.filterwarnings("ignore")

# Set up output directories
output_dir = 'output/satellite-image/pre-processed-images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define date range
start_date = date[0]
end_date = date[1]

# Create a meaningful filename
file_name = f"{start_date[:7]}_{end_date[:7]}"

# Calculate EPSG based on AOI coordinates
lon, lat = aoi.geometries().getInfo()[0]['coordinates'][0][0]
epsg_code = download.convert_wgs_to_utm(lon, lat)

# Split the geometry into smaller grids
grid = geemap.fishnet(aoi, h_interval=0.1, v_interval=0.1, delta=1)
gridList = grid.toList(grid.size())
grid_num = grid.toList(grid.size()).length()

# List to store grid features
ls_feature = [ee.Feature(gridList.get(i)).geometry().bounds() for i in range(grid_num.getInfo())]

# Create temporary grid folder
temp_grid_dir = '/content/temp/grid'
if not os.path.exists(temp_grid_dir):
    os.makedirs(temp_grid_dir)

# Download image by grid
for i, feature in enumerate(ls_feature):
    image = download.Sentinel_no_clouds(feature, start_date, end_date)
    BandIDs = ['B11', 'B8', 'B4', 'B3', 'B2']
    download_id = ee.data.getDownloadId({
        'image': image,
        'bands': BandIDs,
        'region': feature,
        'scale': 10,
        'format': 'GEO_TIFF',
        'crs' : f'EPSG:{epsg_code}',
    })
    response = requests.get(ee.data.makeDownloadUrl(download_id))
    tif_filename = f'/content/temp/grid/image_grid_{i}.tif'
    with open(tif_filename, 'wb') as fd:
        fd.write(response.content)

# Merge grid images
q = os.path.join("/content/temp/grid/image*.tif")
fp = natsorted(glob.glob(q)) 
src_files = [rasterio.open(raster) for raster in fp]
mosaic, out_trans = merge(src_files)

# Set metadata for mosaic
out_meta = src_files[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "count": 5,
    "crs": CRS.from_epsg(int(epsg_code))
})

# Write mosaic raster
temp_main_dir = '/content/temp/main'
if not os.path.exists(temp_main_dir):
    os.makedirs(temp_main_dir)
output_mosaic = os.path.join(temp_main_dir, 'image_snrgb.tif')
with rasterio.open(output_mosaic, "w", **out_meta) as dest:
    dest.write(mosaic.astype(np.float32))

# Clip image to AOI
img_grid = rasterio.open(output_mosaic)
aoi_epsg = aoi.transform(ee.Projection(f'EPSG:{epsg_code}'), 1)
clip, clip_transform = mask(img_grid, aoi_epsg.geometries().getInfo(), crop=True)

# Set metadata for clipped image
out_meta = img_grid.meta.copy()
out_meta.update({
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": 0 and None,
    "height": clip.shape[1],
    "width": clip.shape[2],
    "transform": clip_transform,
    "count": 5,
    "crs": img_grid.crs
})

# Write clipped image
output_clip = os.path.join(output_dir, f'image_snrgb_10m_{file_name}.tif')
with rasterio.open(output_clip, "w", **out_meta) as dest:
    dest.write(clip.astype(np.float32))

# Clean up temporary directories
shutil.rmtree('/content/temp')

# Print download completion status
if img_grid.read().any() == 0:
    print('Error: Images are not available for this area within the given date.')
else:
    print('Download completed!')
#Shoreline extraction
import os
import cv2
import folium
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
from rasterio.enums import Resampling
from rasterio.plot import show_hist
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from codes import shoreline, download, mapping
from codes.shoreline import Create_points
from parameters import aoi, date, horizontal_step, vertical_step
from shapely.ops import unary_union
from scipy import interpolate
from matplotlib import rcParams

# Set up font and formatting settings
plt.rcParams['font.family'] = ['serif']
rcParams['font.stretch'] = 'extra-condensed'

# Create required output folders
output_dirs = [
    'output/satellite-image/post-processed-images',
    'output/shoreline/geojson',
    'output/shoreline/figure/shorelines'
]

for dir_path in output_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Set date range
start_date = date[0]
end_date = date[1]

# Create filename
file_name = f"{start_date[:7]}_{end_date[:7]}"

# Calculate EPSG based on AOI coordinates
lon = aoi.geometries().getInfo()[0]['coordinates'][0][0][0]
lat = aoi.geometries().getInfo()[0]['coordinates'][0][0][1]
epsg_code = download.convert_wgs_to_utm(lon, lat)

# Shoreline extraction
Image = rasterio.open(f'output/satellite-image/pre-processed-images/image_snrgb_10m_{file_name}.tif')
if Image.read().any() == 0:
    print('Warning: The image is empty, so shoreline cannot be extracted.')
else:
    print('Extracting shoreline...')

# ... (rest of your code)

# Save result as GeoJSON file
# ... (rest of your code)

# Create SNB composite
RGB = np.dstack((red, green, blue))

# Adjust the contrast and brightness settings
alpha = 2  # controls the contrast
beta = 0   # controls the brightness
image = np.clip(alpha * RGB + beta, 0, 1)
RGB_transp = image.transpose(2, 0, 1)

# Create plot
fig, ax = plt.subplots(figsize=(7, 7))
show(RGB_transp, ax=ax, transform=transform)
geo_shoreline.plot(ax=ax, facecolor='None', edgecolor='yellow', linewidth=1, label='shoreline')
ax.add_artist(ScaleBar(1, location='lower left', box_alpha=0.5, font_properties={'size': 'small'}))
plt.title(f'{start_date} - {end_date}', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plot_output_path = f'output/shoreline/figure/shorelines/shoreline_{file_name}.png'
plt.savefig(plot_output_path, dpi=300)

# Print completion message
print('Shoreline extraction completed!')
#Analysis
import os
import glob
import natsort
import rasterio
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from shapely.ops import unary_union
from codes.shoreline import Create_points, ExtrapolateOut, ExtrapolateIn
from codes.shoreline import create_union_polygon, create_shoreline_change_points
from codes.shoreline import merge_shoreline_change, linearring_to_polygon
from shapely.geometry import MultiPolygon, Polygon
from parameters import aoi, date
from matplotlib import rcParams

# Set up font and formatting settings
plt.rcParams['font.family'] = ['serif']
rcParams['font.stretch'] = 'extra-condensed'

# Create required output folders
output_dirs = [
    'output/shoreline/retreat&growth',
    'output/shoreline/shoreline-change',
    'output/shoreline/union-shoreline'
]

for dir_path in output_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# File and folder paths
file_path = "input/shoreline"

# Make a search criteria to select the shoreline files
q = os.path.join(file_path, "shoreline_*.json")
shoreline_fp = natsort.natsorted(glob.glob(q))  # sorted files by name

# Import shoreline data
shl_past = gpd.read_file(shoreline_fp[0]).dropna().reset_index(drop=True)
shl_present = gpd.read_file(shoreline_fp[-1]).dropna().reset_index(drop=True)

# Convert LinearRing to Polygon
shl_past = linearring_to_polygon(shl_past)
shl_present = linearring_to_polygon(shl_present)

# Calculate growth and retreat
retreat = gpd.overlay(shl_past, shl_present, how='difference', keep_geom_type=False)
growth = gpd.overlay(shl_present, shl_past, how='difference', keep_geom_type=False)

# Export growth and retreat geometry to GeoJSON
retreat.to_file('output/shoreline/retreat&growth/retreat.json', driver='GeoJSON')
growth.to_file('output/shoreline/retreat&growth/growth.json', driver='GeoJSON')

# ... (rest of your code)

# Create a combination of all shorelines
shape_list = []
for i in range(len(shoreline_fp)):
    shoreline = gpd.read_file(shoreline_fp[i]).dropna().reset_index(drop=True)
    shoreline = linearring_to_polygon(shoreline)
    for k in range(len(shoreline)):
        if isinstance(shoreline['geometry'][k], MultiPolygon):
            polygons = list(shoreline['geometry'][k].geoms)
            shoreline.at[k, 'geometry'] = polygons[0]
    geo_shoreline = unary_union(shoreline['geometry'].exterior)
    shape_list.append(geo_shoreline)

# ... (rest of your code)

# Create one subplot for growth and retreat comparison
fig, ax = plt.subplots(figsize=(7, 7))
show(RGB_transp, ax=ax, transform=transform)
growth.plot(ax=ax, facecolor='blue', label='Growth')
retreat.plot(ax=ax, facecolor='red', label='Retreat')
ax.add_artist(ScaleBar(1, location='lower left', box_alpha=0.5, font_properties={'size': 'small'}))
plt.legend(loc='upper right', fontsize='small')
plt.title(f'{shoreline_fp[0][-18:-5]} - {shoreline_fp[-1][-18:-5]}', fontsize=12)
plt.tight_layout()
plt.savefig('output/shoreline/figure/growth&retreat.png', dpi=300)

# ... (rest of your code)

# Print completion message
if shoreline_fp == []:
    print('Error: There is no shoreline for analysis, and there should be at least 2 shorelines for two different periods.')
else:
    print('Shoreline analysis is finished!')

