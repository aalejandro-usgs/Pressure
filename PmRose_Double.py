#!/usr/bin/env python

from obspy.signal.polarization import particle_motion_odr
from obspy.core import read, UTCDateTime, Stream
from obspy.clients.fdsn.client import Client
import matplotlib.projections as projections
from scipy.signal import coherence
from obspy.io.xseed import Parser
from obspy import read_inventory
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

## User Specificatons ##
debug = True
net = 'IU'
stalocs = ['WCI_00','WCI_10']
stime = UTCDateTime('2018-280T00:00:00.0')
etime = UTCDateTime('2018-310T00:00:00.0')
window = 10.*60.*60.
stepsize= 1.*60.*60.
fmax = 1./10.
fmin = 1./100.
lenfft = 2**14

for pidx, staloc in enumerate(stalocs):
    sta,loc = staloc.split('_')

    ## Read in seismic and pressure data ##
    st = Stream()
    ## WCI has pressure data on 31 instead of 30 ##
    if sta == 'WCI':
        presloc = '31'
    else:
        presloc = '30'

    ctime = stime
    while ctime < etime:

        if net == 'XX':
            string = '/msd/' + net + '_' + sta + '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) + '/' + loc + '_'
            st += read(string + 'LH*')
            st += read('/msd/IU_ANMO/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) + '/' + presloc + '_LDO*')
            ctime += 24.*60.*60.
        else:
            string = '/msd/' + net + '_' + sta + '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) + '/' + loc + '_'
            st += read(string + 'LH*')
            st += read('/msd/' + net + '_' + sta + '/' + str(ctime.year) + '/' + str(ctime.julday).zfill(3) + '/' + presloc + '_LDO*')
            ctime += 24.*60.*60.
    st.merge(fill_value=0)

    ## Read in metadata ##
    sp = Parser()
    if net == 'XX':
        print(st[0].id)
        for tr in st:
            if tr.stats.channel == 'LDO':
                continue
            else:
                stri = tr.id
                inv = read_inventory('/home/aalejandro/Pressure/RESP/RESP.' + stri)
                st.attach_response(inv)
    else:
        client=Client('IRIS')
        inv = client.get_stations(network=net, station=sta, starttime=stime, endtime=etime, channel="LH*", level='response')
    if debug:
        print(inv)

    # Convert to velocity
    st.attach_response(inv)

    # We now have all the data that is either LH or LDO
    if debug:
        print(inv)
        print(st)

    ## Sliding Window ##
    for stT in st.slide(window_length=window, step=stepsize):
        stT.detrend('constant')
        ## Here we grab the pressure and compute the coherence with the two different components ##
        press = stT.select(channel='LDO')[0].data
        if sta == 'WCI':
            press = press*1.43*10**-2 + 8*10**4
        else:
            press = press*0.1
        f,cxy1 = coherence(stT.select(channel='LH1')[0].data, press, nperseg=lenfft)
        f,cxy2 = coherence(stT.select(channel='LH2')[0].data, press, nperseg=lenfft)
        cmean = np.mean(np.array([cxy1[1:],cxy2[1:]]),axis=0)
        f = f[1:]
        cmean = cmean[(f>= fmin) & (f <= fmax)]
        cmean = np.mean(cmean)
        if 'c' not in vars():
            c = cmean
        else:
            c = np.vstack((c, cmean))
        c = np.vstack((c, cmean))
        if net != 'XX':
            stT.rotate(method="->ZNE", inventory=inv)
        stT.filter('bandpass',freqmin=fmin, freqmax=fmax)
        stT.taper(0.05)
        if debug:
            print(stT)
        
        ## Calculating just pressure for colorbar ##
        if 'cpress' not in vars():
            cpress = np.std(press) #Standard deviation in Pa
        else:
            cpress = np.vstack((cpress,np.std(press)))
        cpress = np.vstack((cpress,np.std(press)))

        ## Remove pressure signal and read in azimuth from seismic data ##
        locs = list(set([tr.stats.location for tr in stT]))
        locs.remove(presloc)
        sTT = stT.select(location=loc)
        azi,inc,err_azi,err_inc = particle_motion_odr(sTT)


        ## Calculating particle motion from azimuth data ##
        if 'pm' not in vars():
            pm = np.deg2rad(azi)
        else:
            pm = np.vstack((pm, azi))
        pm = np.vstack((pm + 180., azi))

    ## Limit pressure scale due to gap in WCI data resulting in high non-real pressure values ##
    if sta == 'WCI':
        limit = 529
        pm = pm[cpress < limit]
        c = c[cpress < limit]
        cpress = cpress[cpress < limit]

    pm = np.deg2rad(pm)

    ## Creating figure ##
    fig = plt.figure(1,figsize=(20,10))

    ## Set font parameters ##
    mpl.rc('font',family='serif')
    mpl.rc('font',serif='Times')
    mpl.rc('text', usetex=True)
    mpl.rc('font',size=25)

    ## Plot ##
    if pidx == 0:
        ax = plt.subplot(1,2,1, projection='polar')
        sc = plt.scatter(pm,c, c=cpress, cmap='viridis')
    else:
        ax = plt.subplot(1,2,2, projection='polar')
        sc = plt.scatter(pm,c, c=cpress, cmap='viridis')
    ax.set_theta_zero_location("N")
    ax.set_rlim(0,1)
    ax.set_theta_direction(-1)
    
    ## Lettering ##
    fig.text(0.135, 0.86, '(a)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 30, color='k')
    fig.text(0.600, 0.86, '(b)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 30, color='k')

    del pm
    del c
    del cpress

## Colorbar ##
cbar_ax = fig.add_axes([.18, 0.06, 0.7, 0.02])
cbar = plt.colorbar(sc, fraction=0.06, pad=0.1, orientation='horizontal', cax=cbar_ax)
cbar.ax.set_title('Standard Deviation of Pressure (Pa)')

## Seperating stas for figure titles ##
sta = []
loc = []
for staloc in stalocs:
    staloc = staloc.split('_')
    sta.append(staloc[0])
    loc.append(staloc[1])
plt.subplot(1,2,1)
plt.title(sta[0] + ' ' + loc[0], fontsize=25)
plt.subplot(1,2,2)
plt.title(sta[1] + ' ' + loc[1], fontsize=25)

plt.savefig('PmRose_Double_' + sta[0] + '_' + str(int(1/fmax)) + '-' + str(int(1/fmin)) + 's_' + str(stime.julday) + '-' + str(etime.julday) + '.png')
plt.show()
plt.clf()