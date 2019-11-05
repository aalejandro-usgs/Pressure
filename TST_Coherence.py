#!/usr/bin/env python

from obspy.core import read, Stream, UTCDateTime
from obspy.io.xseed import Parser
from scipy.signal import coherence
from itertools import combinations
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np

## User Specifications ##
debug = True
stime = UTCDateTime('2018-280T00:00:00')
etime = UTCDateTime('2018-310T00:00:00')
net = 'XX'
stalocs = ['TST5_00','TST6_10','TST6_00']
lenfft= 2**14
snpts = 10

## Read in seismic data ##
st = Stream()
for day in range(stime.julday, etime.julday):
    for staloc in stalocs:
        try:
            sta, loc = staloc.split('_')
            st += read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(stime.julday) + '/' + loc + '_LH*')
        except:
            pass

# Remove the vertical
for tr in st.select(channel="LHZ"):
    st.remove(tr)
st.merge(fill_value = 0)
st.sort()
if debug:
    print(st)

## Creating Figure ##
fig = plt.figure(1, figsize=(19,10))
plt.subplots_adjust(hspace=0.0)

## Setting font parameters ##
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=20)

## Plotting ##
for idx, chan in enumerate(['LH1','LH2']):
    plt.subplot(2,1,idx+1)
    stC = (st.select(channel=chan)).copy()
    for comb in combinations(range(len(stC)),2):
        ## Inputs a trace and returns a coherence time series ##
        f,C = coherence(stC[comb[0]].data,stC[comb[1]].data, fs=1., nperseg=lenfft)
        f = f[1:]
        C = C[1:]
        ## Smooth the coherance trace ##
        C = np.convolve(C, np.ones((snpts,))/snpts)[(snpts-1):]
        name1 = (stC[comb[0]].id).replace('.',' ')
        name1 = name1.replace('XX','')
        name2 = (stC[comb[1]].id).replace('.',' ')
        name2 = name2.replace('XX','')
        plt.semilogx(1./f,C, label=name1 + ' and ' + name2)
    plt.legend(ncol=3, loc=8)
    plt.xlim([2.0,10.0**4])
plt.xlabel('Period (s)', fontsize = 25)
plt.subplot(2,1,1)
plt.xticks([])
fig.text(0.060, 0.50, 'Coherence ($\gamma^2$)', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 25)
fig.text(0.085, 0.86, '(a)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 25, color='k')
fig.text(0.085, 0.47, '(b)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 25, color='k')
plt.savefig('TST_Coh.png')
plt.show()