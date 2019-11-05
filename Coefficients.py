#!/usr/bin/env python

from matplotlib.offsetbox import AnchoredText
from obspy.clients.fdsn.client import Client
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
stalocchans = ['ANMO_00_LH1','CCM_10_LH1','ANMO_00_LH2','CCM_10_LH2']
stime = UTCDateTime('2018-280T00:00:00')
etime = UTCDateTime('2018-310T00:00:00')
pmin = 200.
pmax = 2000.

for pidx, stalocchan in enumerate(stalocchans):
	sta,loc,chan = stalocchan.split('_')

	acoeffs = []
	bcoeffs = []
	resis = []

	if sta == 'WCI':
		pressloc = '31'
	else:
		pressloc = '30'

	for day in range(stime.julday, etime.julday):

		### Read in Seismic and Pressure Data ###
		if net == 'XX':
			st = read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + loc + '_' + chan + '.512.seed')
			st += read('/msd/' + net + '_TST5/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + pressloc + '_LDO.512.seed')
		else:
			st = read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + loc + '_' + chan + '.512.seed')
			st += read('/msd/' + net + '_' + sta + '/' + str(stime.year) + '/' + str(day).zfill(3) + '/' + pressloc + '_LDO.512.seed')


		## For rotating data ##
		if net != 'XX':
			client = Client('IRIS')
			inv = client.get_stations(network=net, station=sta, starttime=stime, endtime=etime, channel="LH*", level='response')

		st.detrend('constant')
		st.detrend('linear')
		st.merge(fill_value=0.)
		st.rotate(method="->ZNE", inventory=inv)
		st.filter('bandpass', freqmin=1./pmax, freqmax= 1./pmin)
		st.taper(0.05)
		st.sort()
		if debug:
		    print(st)

		## Import metadata ##
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

		resid = resi([bf[0],bf[1]])/resi([0.,0.])

		acoeffs.append(bf[0])
		bcoeffs.append(bf[1])
		resis.append(resid)

	aSTD = np.std(acoeffs)
	bSTD = np.std(bcoeffs)

	aAVG = np.mean(acoeffs)
	bAVG = np.mean(bcoeffs)

	# if debug:
	# 	print('a: ' + str(acoeffs))
	# 	print('b: ' + str(bcoeffs))
	# 	print('aAVG: ' + str(aAVG))
	# 	print('bAVG: ' + str(bAVG))
	# 	print('resi: ' + str(resis))
	# 	print('aSTD: ' + str(aSTD))
	# 	print('bSTD: ' + str(bSTD))

	## Plot ##

	## Creating figure ##
	fig = plt.figure(1,figsize=(12,13))
	plt.subplots_adjust(wspace=0.075)

	## Set font parameters ##
	mpl.rc('font',family='serif')
	mpl.rc('font',serif='Times')
	mpl.rc('text', usetex=True)
	mpl.rc('font',size=20)

	if pidx == 0:
		plt.subplot(2,2,1)
		plt.plot(bAVG,aAVG,'bD')
		t = np.linspace(0, 2*np.pi, 100)
		plt.plot(bAVG+bSTD*np.cos(t) , aAVG+aSTD*np.sin(t))
		plt.xticks([])
	if pidx == 1:
		plt.subplot(2,2,2)
		plt.plot(bAVG,aAVG,'bD')
		t = np.linspace(0, 2*np.pi, 100)
		plt.plot(bAVG+bSTD*np.cos(t) , aAVG+aSTD*np.sin(t))
		plt.xticks([])
		plt.yticks([])
	if pidx == 2:
		plt.subplot(2,2,3)
		plt.plot(bAVG,aAVG,'bD')
		t = np.linspace(0, 2*np.pi, 100)
		plt.plot(bAVG+bSTD*np.cos(t) , aAVG+aSTD*np.sin(t))
	if pidx == 3:
		plt.subplot(2,2,4)
		plt.plot(bAVG,aAVG,'bD')
		t = np.linspace(0, 2*np.pi, 100)
		plt.yticks([])
		plt.plot(bAVG+bSTD*np.cos(t) , aAVG+aSTD*np.sin(t))
	plt.xlim([-1.01,1.01])
	plt.ylim([-1.01,1.01])
	plt.axvline(0)
	plt.axhline(0)
	sc = plt.scatter(bcoeffs,acoeffs, c=resis, cmap='plasma')
	cbar_ax = fig.add_axes([.935, 0.10, 0.02, 0.78])
	cbar = plt.colorbar(sc, fraction=0.06, pad=0.1, orientation='vertical', cax=cbar_ax)
	plt.clim(0,1.01)
	cbar.ax.set_title('Residual', fontsize=20)

	## Axis labels & Lettering ##
	fig.text(0.06, 0.5, '$\it{a}$' + ' coefficient', ha = 'center', va = 'center', rotation = 'vertical', fontsize = 20)
	fig.text(0.5, 0.045, '$\it{b}$' + ' coefficient', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 20)
	fig.text(0.14, 0.90, '(a)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 20, color='k')
	fig.text(0.55, 0.90, '(b)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 20, color='k')
	fig.text(0.14, 0.48, '(c)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 20, color='k')
	fig.text(0.55, 0.48, '(d)', ha = 'center', va = 'center', rotation = 'horizontal', fontsize = 20, color='k')

## Seperating stas for fig titles ##
sta = []
loc = []
chan = []
for stalocchan in stalocchans:
    stalocchan = stalocchan.split('_')
    sta.append(stalocchan[0])
    loc.append(stalocchan[1])
    chan.append(stalocchan[2])
plt.subplot(2,2,1)
plt.title(sta[0] + ' ' + loc[0] + ' ' + chan[0], fontsize=20)
plt.subplot(2,2,2)
plt.title(sta[1] + ' ' + loc[1] + ' ' + chan[1], fontsize=20)
plt.subplot(2,2,3)
plt.title(sta[2] + ' ' + loc[2] + ' ' + chan[2], fontsize=20)
plt.subplot(2,2,4)
plt.title(sta[3] + ' ' + loc[3] + ' ' + chan[3], fontsize=20)
plt.savefig('Coefficients_' + sta[0] + '.' + loc[0] + '_' + sta[1] + '.' + loc[1] + '.png')
plt.show()
plt.clf()

del bcoeffs
del acoeffs
del bAVG
del aAVG
del bSTD
del aSTD
del resis











