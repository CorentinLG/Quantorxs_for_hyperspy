#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Read and display data recorded by the Scanning Transmission X-ray Microscope (STXM). """

from __future__ import division
import os, re, warnings, itertools, copy
import scipy as sp
import scipy.ndimage as spim
import scipy.signal as sig
import scipy.io as spio
import matplotlib.pyplot as mp
from matplotlib.backends.backend_pdf import PdfPages

__version__ = 20120603
__all__ = ['STXM',
           'Spectra',
           'colours',
           'colors',
           'colour_iter',
           'color_iter',
           '__version__']

# Hex value, HTML name
colours = ( '#FF0000', # red
 			'#000080', # navy 
			'#008000', # green
			'#FF00FF', # fuchsia
			'#3399CC', # turquoise made darker
			'#FF8C00', # dark orange
			'#000000', # black
			'#FFD700', # gold
			'#00FF00', # lime
			'#884513') # saddlebrown
# use colour_iter.next() for the next colour, cycles infinitely
colour_iter = itertools.cycle(colours)

# Damn Americans
colors = colours
color_iter = colour_iter

class STXM(object):
	""" Read header and data, return STXM object.
		
		Usage: import stxm
		       s = stxm.STXM("filename.hdr", align=True, savedlist=True, normalize=True)
		
		If the header is named filename.hdr, then the data is expected
		to be named filename_a.xim (for single image or line scan data),
		or filename_a000.xim, filename_a001.xim, ... for multi-image
		data. The filename may include a path (e.g. dir/subdir/filename.hdr)
		but the data files are expected to be in the same directory as
		the header.
		
		Alternatively, filename.dat may be given. Info will be read from
		filename.dat and data from filename.ncb. If filename.hdr is also
		present in the same directory, additional info not present in .dat
		will be read from .hdr. Reading .dat/.ncb is usually faster than
		reading .hdr/.xim files.
		
		If the header (filename.hdr) describes a multi-image data set
		and an alignment file (filename.aln) is present, then that file
		is read and the alignment is applied to	the data. Set align=False
		to prevent adjusting the data.
		
		If a file list (filename.sl) is stored along with the data, the
		data files and energies to be included are read from the .sl file.
		This can be used e.g. to skip missing or bad data files. Default
		is to read the file if present, set savedlist=False to prevent and
		read all data files. This only works when reading .hdr/.xim, not
		with .dat/.ncb files.
		
		If an Regions Of Interest (ROI) file is present (filename.roi),
		then that file is read and the first defined ROI is assumed to 
		be the I0, the region with 100 % transmission. This is used to
		normalize the data. The data is converted to absorbance: -ln(I/I0).
		The unnormalized data is available under s.data_raw. Set 
		normalize=False to prevent data from being normalized.
				
		The data is stored as s.data[energy, Y, X].
		
		The returned stxm-object can be used to view data, display info,
		or generate images. Type help(stxm) for more help.
	"""
	
	def __init__(self, fname, align=True, savedlist=True, normalize=True,
              read_roi=True):
		""" Automatically called when STXM object is initialized. """
		
		if not os.path.isfile(fname):
			msg = "File {} does not exist or cannot be read.".format(fname)
			raise IOError(msg)
		
		# initialize global variables
		self.header = {}
		self.X_min = 0 # position of stage in microns
		self.X_max = 0
		self.X_px = 0
		self.X_length = 0
		self.X_res = 0
		self.Y_min = 0
		self.Y_max = 0
		self.Y_px = 0
		self.Y_length = 0
		self.Y_res = 0
		self.energy_points = 0
		self.energies = []
		self.regions = 0
		self.infotxt = ''
		self.info = {}
		self.ismultiregion = False
		self.islinescan = False
		self.isabsorbance = False
		self.data = []     # absorbance if ROIs available, otherwise copy of data_raw
		self.data_raw = [] # counts
		self.roi = []      # boolean, True = inside ROI, s.roi[roi_nr, y, x]
		self.roi_labels = []
		self.roifile = None
		self.I0 = []       # if ROIs avaliable
		self.missing_files = []
		self.files = []
		self.original_data = None # stores all data before subset
		self.image_defaults = {
			'cmap': 'gray',
			'vmin': 0,
			'interpolation': 'none',
			'origin': 'lower'
		}
		
		# Read .xim/.hdr files
		if fname.endswith('.hdr'):
			self.basename = os.path.basename(fname[:-4])
			self.basedir = os.path.dirname(fname)
			self.title = self.basename
			
			if savedlist: self._read_savedlist()
			
			self._read_header(fname)
			self._basic_info()
			self._read_data_xim()
		
		# Read .ncb/.dat files
		elif fname.endswith('.dat'):
			self.basename = os.path.basename(fname[:-4])
			self.basedir = os.path.dirname(fname)
			self.title = self.basename
			
			self._read_data_ncb()
			
			# We read .dat file, but use extra info from .hdr if it exists
			if os.path.isfile(os.path.join(self.basedir, self.basename + '.hdr')):
				self._read_header(os.path.join(self.basedir, self.basename + '.hdr'))
				self._basic_info(dat=True)
		else:
			msg = "{} does not end with .hdr or .dat. ".format(fname)
			msg += "Make sure this is a STXM header file."
			raise IOError(msg)
		
		if align: self._read_align()
		
		self.data_raw = self.data.copy()
		if read_roi: self.read_roi()
		if len(self.roi) > 0: self.normalize()
		
		self._info_text()
	
	
	def _read_savedlist(self):
		""" Internal function, read savedlist (.sl) file if present. """
		try:
			# read .sl file
			fh = open(os.path.join(self.basedir, self.basename + '.sl'), 'rU')
			sl = fh.readlines()
			fh.close()
		except IOError:
			# .sl file not present, ignore silently and return
			return
		
		# First line is dir name, skip
		no_files = len(sl[1:])
		self.files = [""] * no_files
		self.energies = [0.0] * no_files
		
		for n, line in enumerate(sl[1:]):
			cell = line.split()
			self.files[n] = os.path.join(self.basedir, cell[0])
			self.energies[n] = float(cell[1])
		
		self.energies = sp.asarray(self.energies)
		self.energy_points = len(self.energies)
	
	
	def _read_header(self, fname):
		""" Internal function, read header file, return dict. """
		# read header file
		fhandle = open(fname, mode='r', encoding='utf-8')
		hdr = fhandle.read()
		fhandle.close()
		
		# Header structure looks a lot like Python dictionary data type, but not quite.
		# Clean up and turn into dict.
		#
		# Replace any space char (line ending, tab etc) and multiples with single space,
		# and remove spaces and ; at end
		hdr = re.sub('\s+', ' ', hdr).strip(' ;')
		# surround every word (letters, numbers, _ and .), surrounded by but not including
		# non-word characters (, ; etc.), with 'xxx'. Or group of words in "x y z" -> '"x y z"'
		hdr = re.sub('([\w.]+)|(?:\".*?\")', '\'\g<0>\'', hdr)
		# remove all double-quotes
		hdr = hdr.replace('"', '')
		# remove ' from numbers, so they can be interpreted as real numbers
		hdr = re.sub('\'([-+\d.eE]+)\'', '\g<1>', hdr)
		# replace ; with , and = with :
		hdr = hdr.replace(';', ',')
		hdr = hdr.replace('=', ':')
		# remove end of sequence commas, prevents unnecessary empty elements
		hdr = re.sub(',\s*(?=[}\)])', '', hdr)
		# surround entire thing in {}
		hdr = "{"+hdr+"}"
		
		# turn string into dict
		self.header = dict(eval(hdr))
	
	
	def _basic_info(self, dat=False):
		""" Internal function, generate some shortcuts to basic info from header. """
		if 'Image' not in self.header['ScanDefinition']['Flags']:
			msg = "Data is not image format (.xim files). "
			msg += "For spectra format (.xsp or .xas files), use Spectra class."
			raise IOError(msg)
		
		self.regions = self.header['ScanDefinition']['Regions'][0]
		if self.regions > 1: self.ismultiregion = True
		
		if 'Line Scan' in self.header['ScanDefinition']['Type']: self.islinescan = True
		
		if not self.ismultiregion:
			if not dat:
				# Don't override what we read from .dat file
				self.X_px = self.header['ScanDefinition']['Regions'][1]['PAxis']['Points'][0]
				self.Y_px = self.header['ScanDefinition']['Regions'][1]['QAxis']['Points'][0]
			
			# X/Y min/max are positions on stage in microns.
			# Not useful here, set to 0 - max
			# In linescan, X is eV min/max. Do keep those.
			self.X_min = self.header['ScanDefinition']['Regions'][1]['PAxis']['Min']
			self.X_max = self.header['ScanDefinition']['Regions'][1]['PAxis']['Max']
			if not self.islinescan:
				self.X_length = sp.absolute(self.X_max - self.X_min)
#				self.X_max = self.X_length
#				self.X_min = 0
				self.X_res = self.X_length*1000/self.X_px
			else:
				self.X_length = sp.absolute(self.X_max - self.X_min)
				self.X_res = self.X_length/self.X_px
			
			self.Y_min = self.header['ScanDefinition']['Regions'][1]['QAxis']['Min']
			self.Y_max = self.header['ScanDefinition']['Regions'][1]['QAxis']['Max']
			self.Y_length = sp.absolute(self.Y_max - self.Y_min) # μm
#			self.Y_min = 0
#			self.Y_max = self.Y_length
			self.Y_res = self.Y_length*1000/self.Y_px # nm/px
		else:
			# find properties for each region and return an array
			self.X_min = sp.zeros(self.regions)
			self.X_max = sp.zeros(self.regions)
			self.X_px = sp.zeros(self.regions)
			self.Y_min = sp.zeros(self.regions)
			self.Y_max = sp.zeros(self.regions)
			self.Y_px = sp.zeros(self.regions)
			for r in range(self.regions):
				self.X_min[r] = self.header['ScanDefinition']['Regions'][r+1]['PAxis']['Min']
				self.X_max[r] = self.header['ScanDefinition']['Regions'][r+1]['PAxis']['Max']
				self.X_px[r]  = self.header['ScanDefinition']['Regions'][r+1]['PAxis']['Points'][0]
				self.Y_min[r] = self.header['ScanDefinition']['Regions'][r+1]['QAxis']['Min']
				self.Y_max[r] = self.header['ScanDefinition']['Regions'][r+1]['QAxis']['Max']
				self.Y_px[r]  = self.header['ScanDefinition']['Regions'][r+1]['QAxis']['Points'][0]
			self.X_length = sp.absolute(self.X_max - self.X_min)
			self.X_min = sp.zeros(self.regions)
			self.X_max = self.X_length
			self.X_res = self.X_length*1000/self.X_px # nm/px, eV/px in line scan
			self.Y_length = sp.absolute(self.Y_max - self.Y_min)
			self.Y_min = sp.zeros(self.regions)
			self.Y_max = self.Y_length
			self.Y_res = self.Y_length*1000/self.Y_px # nm/px
		
		# this is never region dependent
		if self.islinescan:
			# X data is energies, they are not a linear progression
			self.energy_points = self.X_px
			self.energies = sp.array(self.header['ScanDefinition']['Regions'][1]['PAxis']['Points'][1:])
		elif self.energy_points:
			# already set by other means, don't override
			pass
		else:
			self.energy_points = self.header['ScanDefinition']['StackAxis']['Points'][0]
			self.energies = sp.array(self.header['ScanDefinition']['StackAxis']['Points'][1:])
	
	
	def _info_text(self):
		""" Internal function, set infotxt string with current info. """
		# text labels to display info, use join instead of += (faster)
		if self.islinescan:
			if self.basename:
				self.infotxt += "file: {}\n".format(os.path.join(self.basedir, self.basename))
			if self.header and 'ScanDefinition' in self.header.keys() and 'Type' in self.header['ScanDefinition'].keys():
				self.infotxt += "type: {}\n".format(self.header['ScanDefinition']['Type'])
			if self.Y_length:
				self.infotxt += "length: {:.1f} μm\n".format(self.Y_length)
			if self.Y_px:
				self.infotxt += "size: {:d} px\n".format(self.Y_px)
			if self.Y_res:
				self.infotxt += "res: {:.1f} nm/px\n".format(self.Y_res)
			if self.header and 'Time' in self.header.keys():
				self.infotxt += "date: {}\n".format(self.header['Time'])
		else:
			if self.basename: self.infotxt = "file: {}\n".format(os.path.join(self.basedir, self.basename))
			if self.header and 'ScanDefinition' in self.header.keys() and 'Type' in self.header['ScanDefinition'].keys():
				self.infotxt += "type: {}\n".format(self.header['ScanDefinition']['Type'])
			self.infotxt += "energy: {:.1f} eV\n" # fill in later
			if self.ismultiregion:
				for r in range(self.regions):
					self.infotxt += "region {}:\n".format(r)
					if sp.any(self.X_length) and sp.any(self.Y_length):
						self.infotxt += "   length: {:.1f} x {:.1f} μm\n".format(self.X_length[r], self.Y_length[r])
					if sp.any(self.X_px) and sp.any(self.Y_px):
						self.infotxt += "   size: {0:.0f} x {0:.0f} px\n".format(self.X_px[r], self.Y_px[r])
					if sp.any(self.X_res) and sp.any(self.Y_res):
						self.infotxt += "   res: {:.1f} x {:.1f} nm/px\n".format(self.X_res[r], self.Y_res[r])
			else:
				if self.X_length and self.Y_length:
					self.infotxt += "length: {:.1f} x {:.1f} μm\n".format(self.X_length, self.Y_length)
				if self.X_px and self.Y_px:
					self.infotxt += "size: {:d} x {:d} px\n".format(self.X_px, self.Y_px)
				if self.X_res and self.Y_res:
					self.infotxt +=	"res: {:.1f} x {:.1f} nm/px\n".format(self.X_res, self.Y_res)
			if self.header and 'Time' in self.header.keys():
				self.infotxt += "date: {}".format(self.header['Time'])
	
	
	def _read_data_ncb(self):
		""" Internal function, read data from .dat/.ncb files. """
		# ncb files are raw binary, little-endian (<), unsigned (u), 16-bit (2 byte) integers
		self.data = sp.fromfile(os.path.join(self.basedir, self.basename + '.ncb'), dtype=sp.dtype('<u2'))
		self.data = self.data.astype(float)
		
		fh = open(os.path.join(self.basedir, self.basename + '.dat'), 'rU')
		datfile = fh.read().splitlines()
		fh.close()
		
		self.X_px, self.Y_px, scale = datfile[0].split()
		self.X_px = int(self.X_px)
		self.Y_px = int(self.Y_px)
		scale = float(scale)
		self.energy_points = int(datfile[3])
		self.energies = sp.asarray([float(n) for n in datfile[4:4+self.energy_points]])
		
		self.data = self.data.reshape(self.energy_points, self.Y_px, self.X_px) / scale
	
	
	def _read_data_xim(self):
		""" Internal function, read data from .hdr/.xim files. """
		# All possible combo's
		# Type					Flag
		# -----------------------------
		# NEXAFS Point Scan		Spectra
		# OSA Scan				Image
		# Image Scan			Image
		# NEXAFS Line Scan		Image
		# NEXAFS Image Scan		Image Stack
		# 						Multi-Region Image Stack (532_110520027)
		# extra entry ScanType only with NEXAFS Point Scan
		
		# All types of pixel images
		if self.ismultiregion:
			# create array, filled with -1s, large enough to hold largest region
			self.data = -1*sp.ones([self.energy_points, self.regions,
							sp.amax(self.Y_px), sp.amax(self.X_px)], dtype=float)
			
			# now read data
			# What is the filename format of a single energy multi-region image?
			# x_a0.xim, x_a1.xim, or x_a0000.xim, x_a0001.xim
			# assume latter for now
			for en in range(self.energy_points):
				for r in range(self.regions):
					filename = os.path.join(self.basedir, self.basename + '_a{0:03d}{1:d}.xim'.format(en, r))
					try:
						self.data[en, r, 0:self.Y_px[r], 0:self.X_px[r]] = sp.loadtxt(filename)
					except IOError:
						msg = "Data file {} is missing.".format(filename)
						warnings.warn(msg)
		else:
			# one-region image
			if self.islinescan or self.energy_points == 1:
				# single image
				filename = os.path.join(self.basedir, self.basename + '_a.xim')
				try:
					self.data = sp.loadtxt(filename, dtype=float)
				except IOError:
					msg = "Data file {} is missing.".format(filename)
					warnings.warn(msg)
					self.missing = [filename]
					self.data = sp.zeros([self.energy_points, self.Y_px, self.X_px], dtype=float)
			else:
				# multi-image stack
				self.data = sp.zeros([self.energy_points, self.Y_px, self.X_px], dtype=float)
				# Were we given a list of filenames from a .sl file?
				if self.files:
					for en,f in enumerate(self.files):
						try:
							self.data[en] = sp.loadtxt(f)
						except IOError:
							self.missing_files += [f]
					if self.missing_files:
						msg = "The following files were missing despite being "
						msg += "in saved list {}:\n{}".format(os.path.join(self.basedir, self.basename + '.sl'), self.missing_files)
						warnings.warn(msg)
				else:
					missing_idx = []
					for en in range(self.energy_points):
						# files are x_a000.xim, x_a001.xim, ...
						filename = os.path.join(self.basedir, self.basename + '_a{:03d}.xim'.format(en))
						try:
							self.data[en] = sp.loadtxt(filename)
						except IOError:
							missing_idx += [en]
							self.missing_files += [filename]
					
					if len(missing_idx) > 0:
						msg = "There were {} missing data files. ".format(len(missing_idx))
						msg += "Use: print s.missing_files to see which files are missing."
						warnings.warn(msg)
						
						# delete missing data points
						self.data = sp.delete(self.data, missing_idx, axis=0)
						self.energies = sp.delete(self.energies, missing_idx)
						self.energy_points = len(self.energies)
	
	
	def _read_align(self):
		""" Internal function, read alignment file and apply to data. """
		try:
			fh = open(os.path.join(self.basedir, self.basename + '.aln'), 'rU')
			shift = fh.readlines()
			fh.close()
		except IOError:
			# no .aln file, return silently
			return
		
		# Comments start with !, but variable number of lines
		# Then 2 lines with IDL commands
		shift_x = []
		shift_y = []
		for i in shift:
			if not (i.startswith('!') or i.startswith('ALIGN') or i.startswith('PLOTIT')):
				name, x, y = i.split(',')
				shift_x.append(float(x))
				shift_y.append(float(y))
		
		shift = sp.vstack((sp.array(shift_y), sp.array(shift_x))).T
		# shift goes the other way
		shift *= -1
		
		for im in range(self.energy_points):
			self.data[im] = spim.shift(self.data[im], shift[im])
	
	
	def _one_image(self, energy=0, data=[], **kw):
		""" Internal function.
			Sets up a single image based on data; energy is only label.
			Returns figure, image, colorbar, axes, unitlabel, label.
		"""
		# figure object; gets current or create new
		# clear figure, in case interactive
		fig = mp.gcf()
		fig.clf()
		
		# plot position on figure canvas (left, bottom, right, top)
		ax = mp.axes([0.1,0.1,0.6,0.8])
		
		mp.suptitle(self.title)
		mp.minorticks_on()
		
		# Line scan
		if self.islinescan:
			# Are there multi-region line scans? I don't think so.
			# show image, convert axes from pixels to length
			settings = {'extent': [self.X_min, self.X_max, self.Y_min, self.Y_max], 'aspect':4}
			settings.update(self.image_defaults)
			settings.update(kw)
			
			ticks = ax.xaxis.majorTicks
			ticks += ax.yaxis.majorTicks
			ticks += ax.xaxis.minorTicks
			ticks += ax.yaxis.minorTicks
			[t._apply_params(tickdir='out') for t in ticks]
			
			im = mp.imshow(data, **settings)
			
			# Colour scale
			bar = mp.colorbar(shrink=0.8, orientation='horizontal')
			if self.header:
				bar.set_label(self.header['ScanDefinition']['Channels'][1]['Unit'])
			else:
				bar.set_label('counts')
			unit = None
			
			mp.xlabel("Energy (eV)")
			mp.ylabel("Length (μm)")
			
			info = mp.annotate(self.infotxt, xy=(0.725,0.875), xycoords='figure fraction',
							linespacing=1.3, horizontalalignment='left', verticalalignment='top')
			info.draggable()
		else:
			# find range
			if (self.X_length == 0 or self.Y_length == 0 or
				self.X_max == 0 or self.Y_max == 0):
					# If no header was read, we don't have length info
					# to translate pixels to length.
					rng = None
					mp.xlabel("X (px)")
					mp.ylabel("Y (px)")
			else:
				rng = [self.X_min, self.X_max, self.Y_min, self.Y_max]
				mp.xlabel("X (μm)")
				mp.ylabel("Y (μm)")
			
			# image defaults
			settings = {'extent': rng}
			settings.update(self.image_defaults)
			settings.update(kw)
			
			ticks = ax.xaxis.majorTicks
			ticks += ax.yaxis.majorTicks
			ticks += ax.xaxis.minorTicks
			ticks += ax.yaxis.minorTicks
			[t._apply_params(tickdir='out') for t in ticks]
			
			# show image, convert axes from pixels to length, start colour scale from 0
			im = mp.imshow(data, **settings)
						
			# Colour scale and unit
			bar = mp.colorbar(shrink=0.8)
			if self.header and 'ScanDefinition' in self.header.keys():
				unittxt = self.header['ScanDefinition']['Channels'][1]['Unit']
			else:
				unittxt = 'counts'
			unit = mp.annotate(unittxt,	xy=(0.61, 0.85), xycoords='figure fraction')
			unit.draggable()
			
			# display info label
			txt = self.infotxt.format(energy)
			info = mp.annotate(txt, xy=(0.725,0.825), xycoords='figure fraction',
								linespacing=1.3, horizontalalignment='left', verticalalignment='top')
			info.draggable()
		return fig, im, ax, bar, unit, info
	
	
	def read_roi(self):
		""" Read ROI file.
			
			Usage: s.read_rois()		
			
			Reads the ROIs from the filename given in s.roifile, or basename.roi
			by default. The ROI data will be stored as True/False, where
			True means the data included in the ROI. The s.roi
			variable will have the dimensions (roi number, Y px, X px).
			
			Can also read ROIs from IDL .sav files, output from LRNStack
			program. In additions to ROIs, ROI labels will be stored as
			s.roi_labels.
		"""
		if not self.roifile: self.roifile = os.path.join(self.basedir, self.basename + '.roi')
		
		if not os.path.exists(self.roifile):
		      return 0
        
		if self.roifile.endswith('.sav'):
			savfile = spio.readsav(self.roifile)
			self.roi_labels = savfile['specstr']['name'].tolist()
			self.I0 = savfile['i0']
			roidata = savfile['specstr']['inds']
		else:
			try:
				# first line is comment, second gives number of lines in file
				# remove first row (ROI number) and first column (line number)
				# rotate to be row-centric
				roidata = sp.loadtxt(self.roifile, skiprows=2, delimiter=',')[1:, 1:].T
				# missing numbers are given as -1, convert to NaN
				roidata[roidata < 0] = sp.nan
			except IOError:
				msg = "ROI file not found in {}.".format(self.roifile)
				warnings.warn(msg)
				return
				
		# roidata are in pixel number, counting from left bottom corner left-to-right, then up
		# convert to X,Y
		Y = sp.floor_divide(roidata, self.X_px)
		X = sp.remainder(roidata, self.X_px)
		
		# all False
		self.roi = sp.zeros((len(X), self.Y_px, self.X_px), dtype=bool)
		
		for r in range(len(X)):
			# for each ROI, delete NaNs and convert to int
			tmp_X = X[r][~sp.isnan(X[r])].astype(int)
			tmp_Y = Y[r][~sp.isnan(Y[r])].astype(int)
			self.roi[r, tmp_Y, tmp_X] = True
	
	
	def normalize(self):
		""" Convert data to absorbance data.
			
			Usage: s.normalize()
			
			Normalize reads an ROI file through s.read_roi(). Tries to 
			read basename.roi by default, or set an alternative filename to
			s.roifile. The first ROI in the ROI file is assumed to be the I0,
			an area with 100 % transmission. The average background spectrum
			will be saved in s.I0. If an I0 already exists, that one will
			be used instead (user defined I0). If you want to redo I0, 
			set it to empty first (s.I0 = []).
			
			Once ROIs are read the data are converted: absorbance = -ln(data/I0).
			The unnormalized data is stored under s.data_raw
		"""
		if len(self.I0) == 0:
			# Determine I0 from ROIs, read file if not already done; force read if filename given
			if len(self.roi) == 0 or len(self.roifile) > 0: self.read_roi()
			# If still no ROI, stop
			if len(self.roi) == 0:
				msg = "No ROIs loaded, could not normalize data."
				warnings.warn(msg)
				return
			
			# get the I0 always from data_raw
			Y, X = sp.where(self.roi[0])
			self.I0 = self.data_raw[:, Y, X]
			self.I0 = sp.average(self.I0, axis=1)
		
		# reset data if already normalized
		if self.isabsorbance: self.data = self.data_raw.copy()
		
		# mask 0 and negative values with NaN
		self.data[self.data <= 0] = sp.nan
		# add empty dimensions to I0
		i0 = self.I0.reshape(self.energy_points, 1, 1)
		self.data = sp.log(sp.divide(i0, self.data))
		self.data[sp.isnan(self.data)] = 0
		self.isabsorbance = True
	
	
	def subset_data(self, pixels=[], energies=[]):
		""" Subset data.
			
			Usage: s.subset_data(pixels=[X0, Y0, X1, Y1], energies=[low, high])
			
			where 0 is lower left and 1 is upper right. Lower limit is inclusive,
			upper limit is non-inclusive (this is how Python slices). pixels =
			[10,12,15,17] will set the range from (10,12) to (14,16), so a 5 x 5 image.
			Use None to have an open-ended range: pixels = [10, 12, None, None]
			will subset from (10,12) to the rest of the image. Leave empty to not
			subset. For energies, closest match to requested energy will be used.
			
			The original data is stored in s.original_data and is still available
			there, e.g. s.original_data.X_length will give length before subsetting.
			
			Consecutive subsets are possible, but only the first original_data will
			be saved.
		"""
		# check input
		if len(pixels) == 0:
			pixels == [None, None, None, None]
		elif len(pixels) == 4:
			# check range
			if (sp.all(pixels >= 0)) and (pixels[2] <= self.X_px + 1) and (pixels[3] <= self.Y_px + 1):
				if (pixels[0] < pixels[2]) and (pixels[1] < pixels[3]):
					pass
				elif (pixels[0] > pixels[2]) and (pixels[1] > pixels[3]):
					# swap corners
					msg = "Lower left pixel numbers greater than upper right, swapping. "
					msg += "I hope that is what you intended."
					warnings.warn(msg)
					pixels = [pixels[2], pixels[3], pixels[0], pixels[1]]
				else:
					msg = "Define subset as pixels = [X0, Y0, X1, Y1], "
					msg += "where 0 is lower left and 1 is upper right."
					raise ValueError(msg)
			else:
				msg = "Pixels out of range."
				raise ValueError(msg)
		else:
			msg = "Give span of new image as: pixels = [X0, Y0, X1, Y1]. "
			raise ValueError(msg)
		
		# check input
		if len(energies) == 2:
			# sort and find closest match, index
			energies = sp.sort(energies)
			energies = [sp.argmin(sp.absolute(self.energies - n)) for n in energies]
		elif len(energies) == 0:
			energies = [None, None]
		else:
			msg = "Give range of energies as energies = [start, end]. Lower limit is inclusive, "
			msg += "upper limit is non-inclusive (this is how Python slices). "
			msg += "Use None to have an open-ended range: energies = [None, 300] "
			msg += "will limit energies from the lowest to 299. If the exact energy is not found, "
			msg += "the closest match will be used."
			raise ValueError(msg)
		
		# we won't override an existing saved history point
		if not self.original_data:
			self.original_data = copy.deepcopy(self)
		else:
			msg = "A saved object already exists in s.original_data. "
			msg += "I will continue to subset the data as requested, but "
			msg += "resetting will return to the point before the first subset command."
			warnings.warn(msg)
		
		self.data = self.data[energies[0]:energies[1], pixels[1]:pixels[3], pixels[0]:pixels[2]]
		self.X_px = pixels[2] - pixels[0]
		self.Y_px = pixels[3] - pixels[1]
		self.X_min = pixels[0] * self.X_res/1000
		self.Y_min = pixels[1] * self.Y_res/1000
		self.X_max = pixels[2] * self.X_res/1000
		self.Y_max = pixels[3] * self.Y_res/1000
		self.X_length = self.X_max - self.X_min
		self.Y_length = self.Y_max - self.Y_min
		
		self.energies = self.energies[energies[0]:energies[1]]
		self.energy_points = len(self.energies)
		
		self._info_text()
	
	
	def reset_data(self):
		""" Reset the entire data set and linked information back to its original values.
			
			Usage: s = s.reset_data()
			
			!!!! DO NOT FORGET TO ASSIGN THE RETURNED VALUE !!!!
			
			This function does not work in place like all others. The
			STXM object, containing all data and information that was
			saved before the first subset() operation, is returned.
			The original_data variable where it was stored, is then
			cleared. Works only once. If you forget, you loose the data.
		"""
		if self.original_data:
			tmp = copy.deepcopy(self.original_data)
			self.original_data = None
			return tmp
		else:
			warnings.warn('No saved state to return, returning current state.')
			return self
	
	
	def single_image(self, energy=0, data=[], unit='', **kw):
		""" Display a single STXM image.
			
			Usage: s.single_image(energy=0, data=[], unit='')
			
			If the data contains multiple images (a stack), use energy=n (in eV)
			to select the image. Energy=0 (default) will select the first image
			from the stack,	otherwise the closest match to the energy value
			will be selected.
			
			An alternative dataset may be passed. If the alternative data has the
			same shape as the original data, energy may be given to select the
			image. otherwise, data must be a single image and energy will only be
			used for labelling.
			
			The unit for the colourbar can be set, default is 'absorbance' if
			absorbance was calculated, 'counts' otherwise.
			
			All other options are passed through to matplotlib.pyplot.
		"""
		if self.islinescan:
			data = self.data
		elif len(data) > 0:
			# data passed manually
			# make sure data is numpy ndarray
			data = sp.asanyarray(data)
			if len(data) == self.energy_points:
				# image for every energy, slice
				energy_idx = (sp.absolute(self.energies - energy)).argmin()
				energy = self.energies[energy_idx]
				data = data[energy_idx]
			elif data.ndim <= 2 or (data.ndim == 3 and len(data) == 1):
				# single image, energy is label
				pass
			else:
				msg = "Cannot determine which image to show from given data. Either give "
				msg += "a single image or a data set with the same length as the number "
				msg += "of energies."
				raise ValueError(msg)
		else:
			# data from main object
			# only 1 image in data
			if self.energy_points == 1:
				energy = self.energies[0]
				data = self.data
			else:
				# select 1 image from multi-image data
				# find energy closest to requested energy
				energy_idx = (sp.absolute(self.energies - energy)).argmin()
				energy = self.energies[energy_idx]
				data = self.data[energy_idx]
		
		fig, im, ax, bar, bar_label, info = self._one_image(energy=energy, data=data, **kw)
		if len(unit) > 0:
			pass
		elif self.isabsorbance:
			unit = 'absorbance'
		else:
			unit = 'counts'
		bar_label.set_text(unit)
	
	
	def multi_image(self, energy=[], data=[], unit='', tofile=False, **kw):
		""" Display multiple images.
			
			Usage: s.multi_image(energy=[], data=[], tofile=False)
			
			If energy=[] (default), all energies in the stack will be displayed.
			Use energy=[100, 121.8, 122.9, ...] to select a subset of energies to
			display. The closest match to the given energy value will be selected.
			
			If a data set is passed, the energy list will only be used for labels.
			The data must be 3D, with dimensions [energy, X, Y]. All will be imaged.
			
			If tofile is False (default), sends output to the default matplotlib
			backend (usually an on screen device), otherwise saves image to file.
			If output file type is set to PDF (default), then mulitple images will
			be saved into a single PDF file, other image formats will be saved
			as: filebasename_nnn.ext. Use format='png' to change to another output
			format. All other keywords will be passed through to matplotlib.pyplot.
			
			Saving a large number of images can take a long time. To speed things up:
			  >>> import matplotlib
			  >>> matplotlib.use('pdf')
			  >>> import stxm
			
			This makes sure a slower on-screen displaying backend such as Qt4 or
			GTK never gets used.
		"""
		if len(data) > 0:
			# data given
			data = sp.asarray(data)
			if len(energy) > 0:
				# energy also given, use for labels
				if len(energy) != len(data):
					msg = "If you give data and energy, energy will be used as label. "
					msg += "There must be one label for each image."
					raise ValueError(msg)
				else:
					nrg = energy
			else:
				# no energy given, all labels to 0.0 eV
				nrg = [0.0]*len(data)
			nrg_idx = sp.arange(len(nrg))
		else:
			# no data given, use global data
			data = self.data
			
			if len(energy) == 1:
				# don't try to fool the system
				self.single_image(energy[0], tofile=tofile, **kw)
				return
			
			# find closest match to requested energies
			if len(energy) == 0:
				# all of 'em
				nrg_idx = sp.arange(self.energy_points)
				nrg = self.energies
			else:
				nrg_idx = [(sp.absolute(self.energies - en)).argmin() for en in energy]
				nrg = self.energies[nrg_idx]
		
		# PDF is default output format
		if 'format' not in kw: kw['format'] = 'pdf'
		
		# first image
		fig, im, ax, bar, bar_label, info = self._one_image(energy=nrg[0], data=data[0], **kw)
		if len(unit) > 0:
			pass
		elif self.isabsorbance:
			unit = 'absorbance'
		else:
			unit = 'counts'
		bar_label.set_text(unit)
		
		if tofile:
			if kw['format'] == 'pdf':
				pp = PdfPages(os.path.join(self.basedir, self.basename + '.pdf'))
				pp.savefig()
			else:
				mp.savefig(os.path.join(self.basedir, self.basename + "_000." + kw['format']))
		else:
			mp.draw()
			mp.waitforbuttonpress()
		
		# do rest of images
		for n in range(1, len(nrg)):
			# update image
			im.set_data(data[nrg_idx[n]])
			# update colorbar
			bar.set_clim(vmin=0, vmax=sp.amax(data[nrg_idx[n]]))
			bar.draw_all()
			# update text
			info.set_text(self.infotxt.format(nrg[n]))
			
			if tofile:
				if kw['format'] == 'pdf':
					pp.savefig()
				else:
					mp.savefig(os.path.join(self.basedir, self.basename + "_{:03d}.".format(n) + kw['format']))
			else:
				mp.draw()
				# don't wait for last one
				if not n == len(nrg_idx):
					mp.waitforbuttonpress()
		# close file
		if tofile and kw['format'] == 'pdf':
			pp.close()
	
	
	def ratio_image(self, I=0, I0=0, data=[], **kw):
		""" Display an ratio image.
			
			Usage: s.ratio_image(I0=0, I=0)
			
			An absorbance image can be calculated from the ratio of
			two images. This is different from the absorbance data 
			which is normalized to the 100 % transmission region in
			the image, even though both are calculated as an
			absorbance [-ln(I/I0)].
		"""
		# if it is a 2-energy map and no options given,
		# assume lower en = I0, and higher en = I
		if self.energy_points == 2 and I == 0 and I0 == 0:
			I0 = self.energies[0]
			I = self.energies[1]
		
		if (I == 0 or I0 == 0):
			msg = "Give the energies of both I and I0."
			raise ValueError(msg)
		
		# use unnormalized date for ratio
		if len(data) == 0: data = self.data_raw
		
		# find closest match
		I_idx = (sp.absolute(self.energies - I)).argmin()
		I0_idx = (sp.absolute(self.energies - I0)).argmin()
		I = self.energies[I_idx]
		I0 = self.energies[I0_idx]
		
		# mask neg. and zeros
		dt = data[I_idx]
		dt[dt <= 0] = sp.nan
		dt0 = data[I0_idx]
		dt0[dt0 <= 0] = sp.nan
		
		# -ln(I/I0) = ln(I0/I)
		absorbance = sp.log(dt0/dt)
		
		# place nans back with zero
		absorbance = sp.nan_to_num(absorbance)
		
		fig, im, ax, bar, unit, info = self._one_image(data=absorbance, **kw)
		
		# for absorbance images we don't want colour bar to start from 0
		bar.set_clim(vmin=sp.amin(absorbance), vmax=sp.amax(absorbance))
		bar.draw_all()
		
		txt = ''
		if self.basename:
			txt += "file: {}\n".format(os.path.join(self.basedir, self.basename))
		txt += "type: ratio image\n"
		txt += "energy ratio: -ln({:.1f}/{:.1f} eV)\n".format(I, I0)
		if self.X_length and self.Y_length:
			txt += "length: {:.1f} x {:.1f} μm\n".format(self.X_length, self.Y_length)
		if self.X_px and self.Y_px:
			txt += "size: {:d} x {:d} px\n".format(self.X_px, self.Y_px)
		if self.X_res and self.Y_res:
			txt += "res: {:.1f} x {:.1f} nm/px\n".format(self.X_res, self.Y_res)
		if self.header and 'Time' in self.header.keys():
			txt += "date: {}".format(self.header['Time'])
		
		info.set_text(txt)
		unit.set_text("absorbance")
	
	
	def overlay_roi(self, skip=[0]):
		""" Overlay ROIs on existing figures.
			
			Usage: s.overlay_roi(skip=[0])
			
			A Regions Of Interest (ROI) file is read through s.read_roi() and
			the ROI plotted onto the current figure.
			
			ROIs can be skipped by listing their index in the skip=[0] list.
			By default, the first ROI in the file is skipped. The first ROI
			is assumed to be the I0 (100 % transmission) ROI. To display the
			I0 ROI, give an empty list: skip=[]
			
			ROI labels are read from s.roi_labels=["line 1", "hot spot 2"]
			The labels are applied to the ROIs in the order as found in the ROI
			file, but after skipping selected ROIs.
		"""
		# if roifile given, force to read roi again
		if len(self.roi) == 0: self.read_roi()
		# fail if still empty
		if len(self.roi) == 0:
			warnings.warn('No ROIs found in file {}'.format(self.roifile))
			return
		
		# which ROIs to show
		roi_idx = sp.arange(len(self.roi))
		roi_idx = sp.delete(roi_idx, skip)
		
		# determine range
		if (self.X_length == 0 or self.Y_length == 0 or
			self.X_max == 0 or self.Y_max == 0):
				rng = None
		else:
			rng = [self.X_min, self.X_max, self.Y_min, self.Y_max]
		
		# start new colour cycler, so always start from same colour
		citer = itertools.cycle(colours)
		
		for r in roi_idx:
			# draw contour
			cnt = mp.contour(self.roi[r], extent=rng, colors=citer.next(), levels=[0])
			
			if len(self.roi_labels) > 0:
				# neither image nor contour plot is supposed to have legend,
				# hence the unabbreviateness
				cnt.collections[0].set_label(self.roi_labels[r])
		if len(self.roi_labels) > 0:
			l = mp.legend(prop={'size': 'x-small'})
			l.draggable()
	
	
	def set_title(self, title):
		""" Set title for plot.
			
			Usage: s.set_title("Some hilarious title.\\nSecond line.")
			Or:    s.title = "Some hilarious title.\\nSecond line."
			
			By default, the title is set to the filename. If you set the
			title after displaying a figure, you may have to regenerate
			the figure. Use \\n to start a new line. The first form will
			also change the figure in the current figure.
		"""
		self.title = title
		fig = mp.gcf()
		fig.suptitle(title)
	
	
	def show(self):
		""" Display the current image.
			
			Usage: s.display()
			
			If the environment is interactive (e.g. when using the iPython
			shell), updates the current figure. If the environment is not
			interactive (e.g. when running a script), draws the current
			figure on screen, which can then not be altered anymore.
		"""
		if mp.isinteractive():
			# interactive matplotlib session
			mp.draw()
		else:
			# not an interactive matplotlib session
			msg = "Showing an image in a non-interactive environment will "
			msg += "disable any further processing. If you want to save the "
			msg += "image, you must use s.save() *BEFORE* s.show()."
			warnings.warn(msg)
			mp.show()
	
	
	def save(self, filename='', format='pdf', **kw):
		""" Save the current image.
			
			Usage: s.save(filename='', format='pdf')
			
			Saves the current image to a file. The default format is PDF.
			To select another format, give format='...' as parameter. All
			other parameters are passed on to matplotlib's savefig().
			
			The default filename is the name of the header file with the
			format as an extension.
		"""
		if not 'format' in kw.keys(): kw['format'] = format
		if not filename: filename = os.path.join(self.basedir, self.basename + '.' + format)
		mp.savefig(filename, **kw)
	





class Spectra(object):
	""" Read and display .xas and .xsp spectra, produced by aXis2000, or .csv files produced
	 	by LRNStack from STXM data.
		
		Usage: import stxm
		       x = s.Spectra()
		
		The returned Spectra object does nothing but hold a bunch of variables. All variables
		are initialized empty. Start by setting:
			x.files = ['file1.xas', 'file2.xsp', 'file3.csv']
			x.plot_all()
			x.show()
		
		This will plot the given XANES spectra on the default matplotlib backend. xsp Files are
		read, but their header files are ignored. Next, determine at which energy the preedge
		starts:
			x.preedge = [281.0, 395.1]
			x.plot_all()
			x.show()
		
		The preedge data points are fit with a straight line for baseline correction. All
		data points up to the given preedge value are used. If the given value is not an
		exact data point, the closest lower data point is picked. In some cases, there is
		more than one edge in the spectrum (e.g. N and O recorded in one measurement). In
		that case, give a start value (in eV) for that spectrum and 0 for the others:
			x.startat = [0, 380]
		
		This will remove all data points before startat and use only the data points
		between startat and preedge in the correction.
		
		Other parameters that may be set (with examples):
			x.labels = ['nice spectrum', 'even nicer spectrum']  # for legend
			x.scale = [0.5, 2]			# scales spectra
			x.offset = [-0.1, 0.3]		# shift spectra up or down
			x.xlimits = [100, 200]
			x.ylimits = [0, 1.1]
			x.title = "My beautiful plot!"
			x.relative = True			# Y-axis: hide ticks, label: relative absorbance
			x.smooth = [False, True]	# smooth some spectra
			x.smoothtype = 'boxcar'		# see scipy.signal.get_window for more types
			x.smoothwidth = 3
			x.markers = {280.1: 'C=C', 285.6: 'C-C=O'}
		
		The markers are displayed as vertical dashed lines at the given energy, with the
		string value as label. It is very hard to have the labels not overlap, so they are
		positioned in a standard position and made draggable. You can adjust the position
		once displayed. The legend, displayed when labels are present, is also adjustable.
		Note that the markers are given as a Python dictionary with { }. It is also possible
		to use one of the predefined marker sets:
			x.C_markers()
			x.N_markers()
			x.O_markers()
		
		Finally, when done, save the image:
			x.plot_all()  # to redraw
			x.save("filename")
	"""
	def __init__(self):
		""" Automatically run upon initialization. """
		self.files = []
		self.data = []
		self.no_spectra = 0
		self.preedge = []
		self.labels = []
		self.markers = {}
		self.xlimits = []
		self.ylimits = []
		self.startat = []
		self.scale = []
		self.offset = []
		self.title = ""
		self.relative = False
		self.smooth = []
		self.smoothwidth = 3
		self.smoothtype = 'boxcar'
	
	def _read_csv(self, filename):
		""" Read data from csv file(s), exported from LRNStack. """
		try:
			fh = open(filename, 'rU')
		except IOError:
			msg = 'File {} not found.'.format(filename)
			raise IOError(msg)
		l = fh.readline() #ignore 1st
		l = fh.readline()
		this_no_spectra = int(l.split()[0])
		self.no_spectra += this_no_spectra
		self.labels += [fh.readline().split(':', 1)[1].strip() for n in range(this_no_spectra)]
		# read rest as data
		d = sp.loadtxt(fh, delimiter=',', skiprows=2).T
		fh.close()
		
		for i in range(1, this_no_spectra + 1):
			# 1st col is energies, other cols are absorbance data
			# make a 2-col ndarray for each data column, append to data array
			# object array because spectra can have different lengths
			self.data.append(sp.vstack((d[0], d[i])))
	
	def _read_xas(self, filename):
		""" Read data from xas/xsp file, exported from aXis2000. """
		self.data.append(sp.loadtxt(filename, comments="*").T)
		self.no_spectra += 1
		self.labels += ['']
	
	def read_data(self):
		""" Read data from list of files. """
		if len(self.files) == 0:
			msg = "Give data files: x.files = ['file1.xas']"
			warnings.warn(msg)
			return
		
		for f in self.files:
			if f.endswith('.csv'):
				self._read_csv(f)
			elif f.endswith('.xas') or f.endswith('.xsp'):
				self._read_xas(f)
			else:
				msg = 'Warning: {} is not a recognized format. Filename has to '
				msg += 'end with .csv, .xas, or .xsp.'
				warnings.warn(msg)
	
	def plot_all(self):
		""" Does all the hard work. """
		# Test all values, give hints if they are wrong
		if len(self.files) == 0:
			raise ValueError("No files given, try: Spectra.files = ['file1.xas', 'file2.xas']")
		if len(self.preedge) not in [0, self.no_spectra]:
			msg = "There are some preedge numbers given, but not the same number as spectra. "
			msg += "The preedge numbers are used to determine the baseline for the preedge correction. "
			msg += "Unfortunately, this cannot be done automatically, so do this in 3 steps: 1) display "
			msg += "spectra with no preedge given, 2) in the plot determine for each spectrum how many data "
			msg += "points there are before the edge goes up, then 3) enter the preedge numbers and display "
			msg += "again. Give either no preedge numbers (Spectra.preedge = []) or a preedge for each spectrum "
			msg += "(Spectra.preedge = [8, 4, ...])."
			raise ValueError(msg)
		if len(self.labels) not in [0, self.no_spectra]:
			msg = "There are some labels given, but not the same number as spectra. "
			msg += "Give either no labels (Spectra.labels = []) or a label for each spectrum "
			msg += "(Spectra.labels = ['nice spectrum', 'even better spectrum', ...])."
			raise ValueError(msg)
		if len(self.scale) not in [0, self.no_spectra]:
			msg = "There are some scale factors given, but not the same number as spectra. "
			msg += "Give either no scale factors (Spectra.scale = []) or a factor for each spectrum "
			msg += "(Spectra.scale = [0.5, 2, ...])."
			raise ValueError(msg)
		if len(self.offset) not in [0, self.no_spectra]:
			msg = "There are some offsets given, but not the same number as spectra. "
			msg += "Give either no offsets (Spectra.offset = []) or an offset for each spectrum "
			msg += "(Spectra.offset = [0.5, 2, ...])."
			raise ValueError(msg)
		if len(self.markers) != 0:
			try:
				kk = [float(k) for k in self.markers.keys()]
			except ValueError:
				msg = "The markers should be given as a dictionary with the energy position of the "
				msg += "marker as the key and the label as the value: Spectra.markers = {280: 'C=C', 285: 'CO3'}. "
				msg += "Matplotlib uses LaTeX (if installed) for subscripts and superscripts etc, so CO3 "
				msg += "could be given as 'CO$_3$'. To change a value use: Spectra.markers[280] = 'C-C'."
				raise KeyError(msg)
		if len(self.xlimits) not in [0,2]:
			msg = "Either give no xlimits (Spectra.xlimits=[]) or two: lower, upper (Spectra.xlimits=[100,200])."
			raise ValueError(msg)
		if len(self.ylimits) not in [0,2]:
			msg = "Either give no ylimits (Spectra.ylimits=[]) or two: lower, upper (Spectra.xlimits=[0.1,1.2])."
			raise ValueError(msg)
		if len(self.startat) not in [0, self.no_spectra]:
			msg = "If there is a spectrum with more than one edge, and you want to display not the first "
			msg += "edge, then the previous edge will interfere with the baseline correction. To prevent, "
			msg += "give a value in energy units, of where to start (all lower energies will be discarded). "
			msg += "For all spectra where you want the whole spectrum (or at least the first edge), give 0. "
			msg += "So: Spectra.startat = [0, 0, 312.8, 0] if you want to have the 3rd spectrum start at 312.8 "
			msg += "eV (or closest match)."
			raise ValueError(msg)
		if len(self.smooth) not in [0, self.no_spectra]:
			msg = "You have selected smoothing on/off for some files, but not all. "
			msg += "Either give True/False for all spectra or none. Smoothing will use a "
			msg += "boxcar function of width Spectra.smoothwidth (default is 3)."
			raise ValueError(msg)
		
		# All is well, let's do some work
		if self.no_spectra == 0: self.read_data()
		
		# Start with an empty plot
		mp.clf()
		if len(self.smooth) == 0: self.smooth = [False] * self.no_spectra
		for n in range(self.no_spectra):
			x = self.data[n][0]
			y = self.data[n][1]
			if self.smooth[n]:
				w = sig.get_window(self.smoothtype, self.smoothwidth)
				y = sig.convolve(y, w, mode='same')
			if self.startat:
				# determine index of value closest to requested cutoff
				if self.startat[n] == 0:
					s_idx = 0
				else:
					s_idx = sp.argmin(sp.absolute(x - self.startat[n]))
				# slice off unwanted beginning of spectrum
				x = x[s_idx:]
				y = y[s_idx:]
			if self.preedge:
				# determine index of value closest to requested edge
				p = sp.argmax(sp.where(x <= self.preedge[n]))
				# need at least 2 points to fit
				if p < 2: p = 2
				# fit polynomial of order 1
				# a,b are regression parameters
				a,b = sp.polyfit(x[:p], y[:p], 1)
				
				# polyval creates polynomials using coefficients over array of x-vals
				y_base = sp.polyval([a,b], x)
				y -= y_base
			if self.scale:
				# scale spectra if needed
				y *= self.scale[n]
			if self.offset:
				# offset spectra if needed
				y += self.offset[n]
			# any labels
			l = None
			if self.labels:
				l = self.labels[n]
			
			# plot
			mp.plot(x,y, color=colours[n % len(colours)], label=l)
		
		# activate legend, title, limits, labels
		mp.xlabel("energy (eV)")
		mp.minorticks_on()
		
		if self.relative:
			mp.gca().yaxis.set_major_locator(mp.NullLocator())
			mp.gca().yaxis.set_minor_locator(mp.NullLocator())
			mp.ylabel("relative absorbance")
		else:
			mp.ylabel("absorbance")
		
		if self.labels:
			lgd = mp.legend(loc=4, prop={'size': 'x-small'})
			lgd.draggable()
		if len(self.title) > 0: mp.suptitle(self.title)
		if self.xlimits: mp.xlim(self.xlimits)
		if self.ylimits: mp.ylim(self.ylimits)
		
		if self.markers:
			# get a vertical position for the labels
			ymin, ymax = mp.ylim()
			ypos = ymax - 0.1*ymax
			for m in self.markers.keys():
				mp.axvline(x=float(m), linestyle='--', color='black')
				lbl = mp.annotate(self.markers[m], xy=(float(m), ypos), ha='center', backgroundcolor='white')
				lbl.draggable() # move labels around with mouse before saving image
	
	
	def show(self):
		""" Display the current image.
			
			Usage: Spectra.display()
			
			If the environment is interactive (e.g. when using the iPython
			shell), updates the current figure. If the environment is not
			interactive (e.g. when running a script), draws the current
			figure on screen, which can then not be altered anymore.
		"""
		if mp.isinteractive():
			# interactive matplotlib session
			mp.draw()
		else:
			# not an interactive matplotlib session
			mp.show()
	
	
	
	def save(self, filename, format='pdf', **kw):
		""" Save the current image.
			
			Usage: Spectra.save('filename.pdf', format='pdf')
			
			Saves the current image to a file. The default format is PDF.
			To select another format, give format='...' as parameter. All
			other parameters are passed on to matplotlib's savefig().
		"""
		if not 'format' in kw.keys(): kw['format'] = format
		mp.savefig(filename, **kw)
	
	
	def C_markers(self):
		""" Set a standard set of markers for the C edge. """
		#  286.9: 'C$\equiv$N', 286.5: imidazole, 288.5: imidazole (apen, JPhysChem 1993)
		self.markers = {285.0: 'arom. C=C', 286.5: 'C=C-C=O', 288.55: 'RCO$_2$R', 290.3: 'CO$_3$'}
	
	
	def N_markers(self):
		""" Set a standard set of markers for the N edge. """
		# 401.9: 'NHC=O', 402.2: 'C-NH-R', 400.6: imidazole, 401.7: imidazole
		self.markers = {398.6: 'C=N', 399.6: 'C$\equiv$N'}
	
	
	def O_markers(self):
		""" Set a standard set of markers for the C edge. """
		self.markers = {531.2: 'C=O', 532.0: 'RCO$_2$R', 534.4: 'COH\nCOC', 534.9: 'C=C-OH'}



##########################
# Main script
# This is what gets run if this script is run stand-alone
if __name__ == "__main__":
	print(__doc__)
