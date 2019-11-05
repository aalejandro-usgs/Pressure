#!/usr/bin/env python

from obspy.core import read, Stream, UTCDateTime
from obspy.io.xseed import Parser
from obspy.signal import PPSD
import matplotlib.pylab as plt
from scipy.signal import coherence
from scipy.signal import csd
import numpy as np
import glob

## User Specifications ##
debug = True
net = 'IU'
stalocs = ['CCM_10','WCI_10']
year = '2018'
stime = UTCDateTime(year + '-280T00:00:00')
etime = UTCDateTime(year + '-310T00:00:00')
winlen= 10.*60.*60.
winoverlap = 1.*60.*60.
lenfft= 2**14
fmax = 1./200.
fmin = 1./800.

## Setting font parameters ##
import matplotlib as mpl
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
mpl.rc('text', usetex = True)
mpl.rc('font',size=18)


## Function that inputs a trace and returns a coherence time series ##
def coh(tr1,tr2,debug=False):
    cmeans = []
    cstds = []
    times = []
    stime = tr1.stats.starttime
    sT = Stream()
    sT += tr1
    sT += tr2
    for stT in sT.slide(winlen, winoverlap):
        times.append(stT[0].stats.starttime-stime)
        if debug:
            print(stT)
        f,C = coherence(stT[0].data,stT[1].data, fs=1., nperseg=lenfft)
        C = C[(f>= fmin) & (f <= fmax)]
        f = f[(f>= fmin) & (f <= fmax)]
        cmeans.append(np.mean(C))
        cstds.append(np.std(C))
    times = np.asarray(times)
    cmeans = np.asarray(cmeans)
    cstds = np.asarray(cstds)
    return cmeans, cstds, times


## Function for properly indexing subplots ##
def get_index(pidx, idx):
    if (pidx == 0) and (idx == 0):
        return 1
    elif (pidx == 0) and (idx == 1):
        return 3
    elif (pidx == 1) and (idx == 0):
        return 2
    elif (pidx == 1) and (idx == 1):
        return 4


## Creating figure ##
fig = plt.figure(1, figsize=(25,10)) 
plt.subplots_adjust(hspace=0.015, wspace=0.1)

## Defining data for figure ##
for pidx, staloc in enumerate(stalocs):
    sta,loc = staloc.split('_')

    ## Read in data ##
    for day in range(stime.julday, etime.julday):

        if sta == 'WCI':
            Ploc = '31'
        else:
            Ploc = '30'

        if 'st' not in vars():
            st = read('/msd/' + net + '_' + sta + '/' + year + '/' + str(day).zfill(3) + '/' + loc + '_LH*')
            stPr = read('/msd/' + net + '_' + sta + '/' + year + '/' + str(day).zfill(3) + '/' + Ploc + '_LDO*')
        else:
            try:
                st += read('/msd/' + net + '_' + sta + '/' + year + '/' + str(day).zfill(3) + '/' + loc + '_LH*')
                stPr += read('/msd/' + net + '_' + sta + '/' + year + '/' + str(day).zfill(3)
                    + '/' + Ploc + '_LDO*')
            except:
                pass

    ## Remove the vertical and merge & detrend seismic data ##
    for tr in st.select(channel="LHZ"):
        st.remove(tr)
    st.detrend('constant')
    st.merge(fill_value=0.)

    ## Removing the response from seismic data ##
    for tr in st:
        if net == 'XX':
            resp = '/home/aalejandro/Pressure/RESP/RESP.' + tr.id
        else:
            resp = '/APPS/metadata/RESPS/RESP.' + tr.id
        respdic = {'filename': resp, 'date': stime, 'units': 'ACC'}
        tr.simulate(seedresp = respdic)
        tr.data *=10**9 #Change to nm/s^2
        tr.taper(0.05)

    ## Merge, detrend, amd filter pressure data ##
    stPr.merge(fill_value=0)
    stPr.detrend('constant')
    for tr in stPr:
        tr.data = tr.data.astype(np.float32)
        tr.data *=.1
    trP = stPr[0].copy()
    trP.filter('bandpass', freqmin=fmin, freqmax=fmax)
    trP.taper(max_percentage=0.05, side='left')

    ## Futher trim data to specified length of time if desired ##
    stSP = st.copy()
    stSP.merge(fill_value=0)
    st.trim(stime,etime)
    stSP.trim(stime,etime)
    stPr.trim(stime,etime)

    ## Define time for plots ##
    t = np.asarray(range(trP.stats.npts))/(60.*60.*24)

    ## Sort data ##
    stSP.sort()
    stPr.sort()

    ## Plot using indexes defined above ##
    for idx1, tr1 in enumerate(stSP):
        cmeans, cstds, times = coh(tr1,stPr)
        newidx = get_index(pidx, idx1)
        
        tr1.filter('bandpass', freqmin=fmin, freqmax=fmax)
        tr1.taper(0.05)

        LHid = (tr1.id).replace('IU','')
        LHid = (LHid).replace('.',' ')
        Pid = (trP.id).replace('IU','')
        Pid = (Pid).replace('.',' ')

        plt.subplot(3,2,newidx)
        plt.plot(t, tr1.data, 'k', label=LHid)
        plt.plot(t, trP.data, 'r', label=Pid, alpha=0.5)
        plt.xticks([])
        plt.xlim((min(t),max(times/(60.*60.*24))))
        plt.legend(loc=8, ncol=2)

        if pidx == 0:
            plt.subplot(3,2,5)
            plt.ylim([0,1])
        else:
            plt.subplot(3,2,6)
            plt.ylim([0,1])
        
        plt.plot(times/(60.*60.*24.), cmeans, label=LHid)
        
        plt.xlim(min(times/(60.*60.*24.)),max(times/(60.*60.*24.)))
        plt.xlabel('Time (Days)')
        plt.legend(loc=8, ncol=2)

        ## y-axis labels ##
        fig.text(0.087, 0.73, 'Pressure (Pa)', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 20, color='r')
        fig.text(0.087, 0.52, 'Acceleration $(nm/s^2$)', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 20, color='k')
        fig.text(0.087, 0.24, 'Coherence ($\gamma^2$)', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 20)

        ## Lettering ##
        fig.text(0.135, 0.86, '(a)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
        fig.text(0.135, 0.60, '(c)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
        fig.text(0.135, 0.34, '(e)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
        fig.text(0.541, 0.86, '(b)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
        fig.text(0.541, 0.60, '(d)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
        fig.text(0.541, 0.34, '(f)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 22, color='k')
    
    del st, stPr

## Seperating stas for fig titles ##
sta = []
loc = []
for staloc in stalocs:
    staloc = staloc.split('_')
    sta.append(staloc[0])
    loc.append(staloc[1])

plt.savefig('Time_Coh_' + sta[0] + '_' + sta[1] + '.png')
plt.show()