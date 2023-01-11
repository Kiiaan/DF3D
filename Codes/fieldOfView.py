import os, re
import xarray as xr, numpy as np, pandas as pd
import sparse_decoding as spd
from skimage.io import imread
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

class FOV:
    def __init__(self, fov_name, fov_dir, file_regex, rounds, channels, 
                 normalize_max='max', min_cutoff=0, imgfilter=None, smooth_param=None):
        """ normalize_max: if 'max', then each 3D stack is normalized (clipped and scale to 1) 
                            by its maximum value.
                           if a scalar, then each 3D stack is normalized by normalize_max
                           if None, then no normalization
            normalize_min: each image is clipped from below by normalize_min. 0 will not change anything
            imgfilter: str or callable. If str, could either be "gaussian" or "median" with parameter (sigma or window size)
                        in smooth_param . If callable, it should apply to individual 2D images. If None, no
                        filter is applied
        """
        self.normalize_max = normalize_max
        self.min_cutoff = min_cutoff
        self.name = fov_name
        self.fov_dir = fov_dir
        self.regex = file_regex
        self.rnds = rounds
        self.chans = channels
        
        # read images
        self.img_list = self.readFromFile(self.fov_dir, self.regex, self.rnds, self.chans)

        # normalize
        if not self.normalize_max is None:
            newlist = deepcopy(self.img_list)
            for rnd_imgs in newlist:
                for ch_imgs in rnd_imgs:
                    if self.normalize_max == 'max':
                        i_max = np.array(ch_imgs).max()
                    else:
                        i_max = self.normalize_max
                    i_min = self.min_cutoff
                    for i in range(len(ch_imgs)):
                        ch_imgs[i] = (np.clip(ch_imgs[i], i_min, i_max) - i_min) / (i_max - i_min) # no saturation from the top
            self.img_list = newlist
        
        # filtering
        if not imgfilter is None:
            if isinstance(imgfilter, str): 
                if imgfilter == 'gaussian':
                    filt = lambda x: gaussian_filter(x, smooth_param)
                elif imgfilter == 'median':
                    filt = lambda x: median_filter(x, smooth_param)
                else:
                    raise ValueError('Image filter can only be "gaussian" or "median" or a callable')
            else:
                filt = imgfilter
            newlist = deepcopy(self.img_list)
            for rnd_imgs in newlist:
                for ch_imgs in rnd_imgs:
                    for i in range(len(ch_imgs)):
                        ch_imgs[i] = filt(ch_imgs[i])
            self.img_list = newlist
            
    def get_xr(self):
        """Reads 5-dimensional data and stores it in an xarray with these coordinates:
            RND, CH, z, y, x"""
        fov_stack = xr.DataArray(self.img_list, dims = (spd.RND, spd.CHN, 'z', 'y', 'x'), 
                                 coords = {spd.RND:range(len(self.rnds)), spd.CHN:range(len(self.chans))})
        return(fov_stack)
        
    @staticmethod
    def readFromFile(fdir, fregex, rnds, chans):
        """ Reads 5-dimensional data and stores it in a list of lists: RND-CH-z-y-x"""
        return [FOV.read4D_rnd(fdir, rnd, fregex, chans) for rnd in rnds]
        
    @staticmethod
    def read3D_rnd_ch(imgdir, rnd, ch, file_regex):
        files = []
        for f in sorted(os.listdir(imgdir)):
            mtch = file_regex.match(f)
            if mtch is not None:
                if (mtch.group('rndName') == rnd) and (mtch.group('ch') == ch):
                    files.append(os.path.join(imgdir, f))
        files = sorted(files)
        return [imread(f) for f in files]

    @staticmethod
    def read4D_rnd(imgdir, rnd, file_regex, channels):
        return [FOV.read3D_rnd_ch(imgdir, rnd, ch, file_regex) for ch in channels]

    def mip(self):
        """ Maximum intensity projection over z-axis"""
        a = self.get_xr()
        amip = a.reduce(np.max, dim='z', keepdims=True)
        return amip
        
    def samplePixels(self, size=None, sample_frac=None, min_norm=0, is2D = True):
        """ Takes a random sample of pixels whose norm is greater than min_norm. 
            size: Maximum sample size. If set, the sample size will be min(size, #eligible pixels)
            sample_frac: If set, the sample size will be set by the sample_frac fraction of eligible pixels
            is2D: If true, then the intensities are maximum projected before sampling. 

            Output is a DataArray with dims (RNDCH, spatial)
        """
        if is2D:
            ints = self.mip()
        else:
            ints = self.get_xr()
            
        if ints.dims == (spd.RND, spd.CHN, 'y', 'x'):
            ints_flat = ints.stack(spatial=['y', 'x']).stack(RNDCH=[spd.RND, spd.CHN]).transpose('spatial', 'RNDCH')
        else:
            ints_flat = ints.stack(spatial=['z', 'y', 'x']).stack(RNDCH=[spd.RND, spd.CHN]).transpose('spatial', 'RNDCH')

        norms = np.linalg.norm(ints_flat.values, ord=2, axis=1)
        ints_flat = ints_flat[norms > min_norm]
        if size is None and sample_frac is None:
            sample_size = ints_flat.shape[0]
        if not size is None: 
            sample_size = np.minimum(ints_flat.shape[0], size)
        if not sample_frac is None:
            sample_size = int(ints_flat.shape[0] * sample_frac)
        samp_inds = np.random.choice(range(ints_flat.shape[0]), 
                                     size=int(sample_size),
                                    replace=False)
        return ints_flat[samp_inds]