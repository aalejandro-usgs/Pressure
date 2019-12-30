#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def setupmap(central_lon, central_lat,handle):
    #handle = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    handle.set_extent(extent)

    handle.add_feature(cfeature.LAND)
    handle.add_feature(cfeature.OCEAN)
    handle.add_feature(cfeature.COASTLINE)
    handle.add_feature(cfeature.BORDERS, linestyle=':')
    handle.add_feature(cfeature.LAKES, alpha=0.5)
    handle.add_feature(cfeature.RIVERS)
    handle.add_feature(cfeature.STATES, edgecolor='gray')
    return handle


# I think your HRV coordinate is wrong
lats = [34.945911,38.055698,42.506401,34.945911,32.309799,38.228901]
lons = [-106.457199,-91.244598,-71.558296,-106.457199,-110.784698,-86.2939]

fig= plt.figure(figsize=(12,12))

boxcoords=[20., -120, 55 ,-66.]
extent=[boxcoords[1], boxcoords[3], boxcoords[0], boxcoords[2]]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])            
ax = plt.subplot(1,1,1, projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax = setupmap(central_lon, central_lat, ax)
sc = ax.scatter(lons, lats, transform=ccrs.PlateCarree(),marker='D',color='k')
ax.stock_img()
plt.savefig('Station_Map.png', format='PNG')
plt.show()
