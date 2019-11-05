#!/usr/bin/env python

from matplotlib.offsetbox import AnchoredText
from scipy.signal import coherence, hilbert
from obspy.core import read, UTCDateTime
from obspy.signal.invsim import evalresp
from obspy.io.xseed import Parser
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from matplotlib.mlab import csd
import matplotlib as mpl
import numpy as np
import sys


debug = True
net = 'IU'
stas = ['ANMO','CCM','HRV','TUC','WCI']
locs = ['00','10']
chans = ['LH1','LH2']
stime = UTCDateTime('2018-280T00:00:00')
etime = UTCDateTime('2018-310T00:00:00')
Pmin = 200.
Pmax = 2000.
NFFT = 2**14
            
for sta in stas:
	for loc in locs:
		for chan in chans:

			if sta == 'WCI':
				pressloc = '31'
			else:
				pressloc = '30'
				if net == 'XX':
					staLDO = "TST5"
				else:
					staLDO = sta

			## Create .csv file ##
			myfile = open('Table2_data_' + loc + '_' + sta + '_' + chan + '.csv', 'w')
			myfile.write('Station, Day, a coeff, b coeff, Residual, Coherence \n')

			for day in range(stime.julday, etime.julday):

				### Read in Seismic and Pressure Data ###
				st = read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + loc + '_' + chan + '.512.seed')
				st += read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + pressloc + '_LDO.512.seed')

				st.detrend('constant')
				st.detrend('linear')
				st.merge(fill_value=0.)
				if debug:
				    print(st)

				## Calculate coherence ##
				f, Cxy  = coherence(st[0].data, st[1].data, fs=1., nperseg=NFFT)
				CM = np.mean(Cxy[(f>= 1./Pmax) &(f<= 1./Pmin)])
				tr = st.select(channel=chan)[0]
				freq = csd(tr.data, tr.data, NFFT= NFFT, noverlap = int(0.5*NFFT), Fs = 1./tr.stats.delta, scale_by_freq=True)

				## Remove 0 frequency point ##
				freq = freq[1:]


				## Import metadata ##
				for tr in st.select(channel='LH*'):
				    if net == 'XX':
				        resppath = '/home/aalejandro/Pressure/RESP/RESP.' + tr.id
				    else:
				        resppath = '/APPS/metadata/RESPS/RESP.' + tr.id
				    if debug:
					    print(resppath)
				resp = evalresp(t_samp = tr.stats.delta, nfft=NFFT, filename=resppath, date = tr.stats.starttime, station = tr.stats.station, channel= tr.stats.channel,
				                locid = tr.stats.location, network = tr.stats.network, units = "ACC")



				st.filter('bandpass', freqmin=1./Pmax, freqmax= 1./Pmin)
				st.taper(0.05)
				st.sort()


				sp = Parser()
				for tr in st.select(channel='LH*'):
				    if net == 'XX':
				        sp.read('/home/aalejandro/Pressure/RESP/RESP.' + net + '.' + sta + '.' + loc + '.' + chan)
				    else:
				        sp.read('/APPS/metadata/RESPS/RESP.' + net + '.' + sta + '.' + loc + '.' + chan)
				paz = sp.get_paz(net + '.' + sta + '.' + loc + '.' + chan, stime)


				### Convolve Pressure Signal with Seismometer Response ###
				st.select(channel="LDO").simulate(paz_simulate=paz)
				st.normalize()

				### Pressure Corrected Signal Function ###
				def presscorrt(x):
				    st2 = (st.select(channel=chan)).copy()
				    return st2[0].data - x[0]*st.select(channel="LDO")[0].data - x[1]*np.imag(hilbert(st.select(channel="LDO")[0].data))   
				def resi(x):
				    val = np.sum(presscorrt(x)**2)/len(presscorrt(x))
				    return val

				### Hilbert Transform of convolved pressure ###
				b = -np.imag(hilbert(st.select(channel='LDO')[0].data))
				### Minimizing seismic function / a and b coeffs for fmin ###
				bf = fmin(resi, [0.,0.])
				times = np.arange(st[0].stats.npts)/(60.*60.)

				## Output data to .csv file ##
				myfile.write(sta + '_' + loc + '_' + chan + ', ' + str(day).zfill(3) + ', ' + str(bf[0]) + ', ' + str(bf[1]) + ', ' + str(resi([bf[0],bf[1]])/resi([0.,0.]))
				    + ', ' + str(CM) + '\n')\

			myfile.close()






