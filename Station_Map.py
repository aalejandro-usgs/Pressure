#!/usr/bin/env python

from obspy.imaging.maps import plot_basemap
from mpl_toolkits.basemap import Basemap
from obspy.clients.fdsn import Client
from matplotlib.cm import get_cmap
from obspy.core import UTCDateTime
from obspy import read_inventory 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

## Set Font Parameters ##
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

debug = True
sta = []
lat = []
lon = []

# Grab the stations lats and lons ##
f = open('Fig1_stations.csv','r')
info = f.readline()
for line in f:
    line = line.split(',')
    sta.append(str(line[0]))
    lat.append(float(line[1]))
    lon.append(float(line[2]))
f.close()
if debug:
	print('sta: ' + str(sta))
	print('lat: ' + str(lat))
	print('lon: ' + str(lon))

## Create Figure ##
plt.figure(1, figsize=(12,12))
m = Basemap(width=5500000,height=2858550,projection='lcc',
            resolution='c',lat_0=39,lon_0=-97.)

## Plot ##
m.drawcountries()
m.shadedrelief()
x,y = m(lon, lat)
m.scatter(x,y,marker='D',color='k')
for tri in zip(x,y,sta):
	if tri[2] == 'TST':
		plt.text(tri[0]-100000,tri[1]-150000, tri[2], fontsize=13)
	else:
		plt.text(tri[0]-100000,tri[1]+70000, tri[2], fontsize=13)

plt.savefig('Station_Map.PNG')
plt.show()