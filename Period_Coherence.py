#!/usr/bin/env python

from obspy.core import UTCDateTime, read, Stream
from scipy.signal import coherence
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib as mpl
import numpy as np

## User Specifications ##
debug = True
net ='IU'
stalocs = ['ANMO_00','CCM_10']
chan1 = 'LH1'
chan2 ='LDO'
stime = UTCDateTime('2018-280T00:00:00')
etime = UTCDateTime('2018-310T00:00:00')
winlen= 10.*60.*60.
winoverlap = 1.*60.*60.
lenfft= 2**14


## Creating figure ##
fig = plt.figure(1,figsize=(24,12))
plt.subplots_adjust(wspace=0.1)

## Font parameters ##
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=22.5)

for pidx, staloc in enumerate(stalocs):
    sta,loc1 = staloc.split('_')

    if sta == 'WCI':
        loc2 = '31'
    else:
        loc2 = '30'

    ## Read in seismic and pressure data ##
    st = Stream()
    for day in range(stime.julday,etime.julday):
        st += read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + loc1 + '_' + chan1 + '*')
        st += read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + loc2 + '_' + chan2 + '*')

    st.detrend('constant')
    st.merge(fill_value=0.)
    if debug:
        print(st)

    ## Sliding Window ##
    for stT in st.slide(window_length=winlen, step=winoverlap):
        f,cxy = coherence(stT[0].data, stT[1].data, nperseg= lenfft)
        cxy = cxy[1:]
        f = f[1:]
        # if variable does not exist
        if 'c' not in vars():
            c=cxy
        else:
            # if variable exists
            c = np.vstack((c, cxy))


    ## Make a vector of times from the number of coherence calculations ##
    t = range(len(c[:,1]))
    t = [x/24. for x in t] #Change from hours to days

    ## Plot ##
    if pidx ==0:
        plt.subplot(1,2,1)
        plt.pcolormesh(1./f,t,c)
    else:
        plt.subplot(1,2,2)
        plt.pcolormesh(1./f,t,c)
        plt.yticks([])

    plt.xlabel('Period (s)')
    plt.xscale('log')
    plt.xlim((0.,4000.))

    ## Axis labels & Lettering ##
    fig.text(0.09, 0.5, 'Time (Days)', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 25)
    fig.text(0.110, 0.86, '(a)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 32, color='k')
    fig.text(0.515, 0.86, '(b)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 32, color='k')

    del c

## Colorbar ##
cbar_ax = fig.add_axes([.935, 0.15, 0.02, 0.7])
cbar = plt.colorbar(fraction=0.06, pad=0.1, orientation='vertical', cax=cbar_ax)
cbar.set_clim((0.0, 1.0))
cbar.ax.set_title('Coherence ($\gamma^2$)', fontsize=22)

## Seperating stas for fig titles ##
sta = []
loc1 = []
for staloc in stalocs:
    staloc = staloc.split('_')
    sta.append(staloc[0])
    loc1.append(staloc[1])
plt.subplot(1,2,1)
plt.title(sta[0] + ' ' + loc1[0] + ' ' + chan1, fontsize=25)
plt.subplot(1,2,2)
plt.title(sta[1] + ' ' + loc1[1] + ' ' + chan1, fontsize=25)

plt.savefig('Period_Coh_' + sta[0] + '_' + sta[1] + '.png')
plt.show()