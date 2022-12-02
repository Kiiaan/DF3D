import os, re
import argparse, yaml
import xarray as xr, numpy as np, pandas as pd
import sparse_decoding as spd
from skimage.io import imread
from copy import deepcopy
from functools import reduce as ftreduce, partial
from sklearn.linear_model import LinearRegression
import time 
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
import logging
from joblib import Parallel, delayed
from datetime import datetime

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
        self.img_list = self.readFromFile(self.fov_dir, self.regex)

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
                        ch_imgs[i] = (np.clip(ch_imgs[i], i_min, i_max)) / (i_max) # no saturation from the top
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
    def readFromFile(fdir, fregex):
        """ Reads 5-dimensional data and stores it in a list of lists: RND-CH-z-y-x"""
        return [FOV.read4D_rnd(fdir, rnd, fregex) for rnd in rnds]
        
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
    def read4D_rnd(imgdir, rnd, file_regex):
        return [FOV.read3D_rnd_ch(imgdir, rnd, ch, file_regex) for ch in channels]

    def mip(self):
        """ Maximum intensity projection over z-axis"""
        a = self.get_xr()
        amip = a.reduce(np.max, dim='z', keepdims=False)
        return amip
        
    def samplePixels(self, sample_frac=1, min_norm=0, is2D = True):
        """ Samples sample_frac fraction of pixels whose norm is greater than min_norm
            Output is a DataArray with dims (RNDCH, spatial)
            If is2D, then the intensities are maximum projected before sampling. 
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
        samp_inds = np.random.choice(range(ints_flat.shape[0]), 
                                     size=int(ints_flat.shape[0] * sample_frac),
                                    replace=False)
        return ints_flat[samp_inds]
        
def deconv(int_xarr, codebook, size=None, min_norm=0.3, alpha=0.02):
    dcObj_ = spd.Decoder2D(int_xarr, codebook, alpha=alpha, size=size)
    dcObj_.prepTrainingPixels(min_norm=min_norm)
    dcObj_.applyLasso()
    dcObj_.applyOLS()
    return dcObj_

def normAndDeconv(int_xarr, codebook, min_norm=0.3, alpha=0.02, chanCoefs=None, size=None):
    """ Normalize intensities by chanCoefs, then decode
        int_xarr: Intensity data array, either with dims (RNDCH, spatial) or (RND, CHN, y, x)
        chanCoefs: a numpy vector. If None, then will be set to a vector of ones
    """
    # flattening the images. It's not necessary but makes the code slightly more readable
    if int_xarr.dims == (spd.RND, spd.CHN, 'y', 'x'):
        size = (int_xarr['x'].shape[0], int_xarr['y'].shape[0])
        int_xarr = int_xarr.stack(spatial=['y', 'x']).stack(RNDCH=[spd.RND, spd.CHN]).transpose('spatial', 'RNDCH')
     
    # make sure chanCoefs is a row vector
    if chanCoefs is None:
        chanCoefs = np.ones((1, int_xarr.shape[1]))
    else:
        chanCoefs = np.array(chanCoefs).reshape((1, -1))

    intensities = int_xarr / chanCoefs # normalize
    dcObj = deconv(intensities, codebook, min_norm=min_norm, alpha=alpha, size=size)
    return dcObj    

def estimateChannelCoefs(int_arr, codebook, min_norm=0.3, alpha=0.02, n_iter=3):
    """ int_arr: Intensity data array, either with dims (RNDCH, spatial) or (RND, CHN, y, x)"""
    print("Channel estimation with data array size {}".format(int_arr.shape))
    cb_flat = codebook.stack(flatcode = (spd.RND, spd.CHN))
    coefs_list = [np.ones(cb_flat.shape[1])]

    for n in range(n_iter):
        intensities_ = deepcopy(int_arr)
        coefs = ftreduce(lambda a, b: a*b, coefs_list)
        t1= time.time()
        dcObj_ = normAndDeconv(intensities_, cb, min_norm=min_norm, alpha=alpha, chanCoefs=coefs)
        t2 = time.time()
#         coefs_rnch = np.reshape(coefs, (1, -1))
#         intensities_ = intensities_ / coefs_rnch
#         dcObj_ = deconv(intensities_, cb, min_norm=min_norm, alpha=alpha)
        print("Decoding took {} seconds".format(t2 - t1))

        """ Finding the coefficients with the new decoding results"""
        sumWeights = (cb_flat.values.T[:, None, :] * dcObj_.ols_table.values.T).sum(axis=-1) # summing the weights in all pixels, in all cycles that the spots are "on"

        sumWeights = xr.DataArray(sumWeights, dims=('flatcode', 'pixels'), 
                                 coords={'rndch': ('flatcode', cb_flat.coords['flatcode'].values),
                                            spd.CHN : ('flatcode', cb_flat.coords[spd.CHN].values),
                                            spd.RND : ('flatcode', cb_flat.coords[spd.RND].values),
                                            'x' : ('pixels', dcObj_.ols_table.coords['x'].values),
                                            'y' : ('pixels', dcObj_.ols_table.coords['y'].values)})

        int_flat_ = dcObj_.ols_pixs.transpose()

        lr = LinearRegression(fit_intercept=False)
        newcoefs = []
        for i in range(sumWeights.shape[0]):
            x = sumWeights[i].values
            y = int_flat_[i].values
            y = y[x > 0.1]
            x = x[x > 0.1].reshape(-1, 1)
            newcoefs.append(lr.fit(x, y).coef_[0])
        newcoefs = np.array(newcoefs)
        coefs_list.append(newcoefs)
        t3 = time.time()
        print("Estimating the channel coefficients took {} seconds".format(t3 - t2))
    coefs_list = [ftreduce(lambda a, b: a*b, coefs_list[:(i+1)]) for i in range(len(coefs_list))]
    coefs_df = pd.DataFrame(coefs_list).T
    coefs_df.columns = ["iter{}".format(i) for i in range(len(coefs_df.columns))]

    return coefs_df



parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

in_dir = params['reg_dir']
out_dir = params['dc_out']
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

suffix = params['dc_suff']

rnds = params['dc_rounds']
channels = params['channel_order'] 
dc_npool = params['dc_npool']

pat3d = r"(?P<rndName>\S+)?_(?P<fov>FOV\d+)_(?P<z>z\d+)_(?P<ch>ch\d+)\S*.tif"
regex_3d = re.compile(pat3d)
fov_pat = params['fov_pat']
fov_names = sorted([f for f in os.listdir(in_dir) if re.match(fov_pat, f)])

min_intensity = params['min_intensity']
normalize_ceiling = params['max_intensity'] # set to None in order to not normalize the maximum of each image
smooth_method = params['smooth_method'] # smoothing method
sOrW = params['smooth_degree'] #sigma for gaussian blur OR size (width) for median

cb_file = params['codebook_path']
cb = spd.Codebook.readFromFile(cb_file)

min_dc_norm = params['min_dc_norm'] # minimum pixel norm for decoding
lasso_alpha = params['lasso_reg_value'] # the regularization value for the lasso model
weight_thresh = params['deconv_weight_threshold'] # the hard threshold on the deconvoled weights from the OLS model

chanCoef_fovs = params['chan_coef_fovs'] # if list, name of fovs to use for channel coef estimation. If int, number of fovs to sample
chanCoef_frac = params['chan_coef_frac'] # fraction of pixels to sample from each fov to estimate channel coefficients
chanCoef_iter = params['chan_coef_niter']
chanCoef_file = os.path.join(out_dir, "channel_coefficients{}.tsv".format(suffix)) # params['chan_coef_file']
chanCoef_plot = os.path.join(out_dir, "channel_coefficients{}.pdf".format(suffix)) # params['chan_coef_file']

if isinstance(chanCoef_fovs, int):
    chanCoef_fovs = list(np.random.choice(fov_names, chanCoef_fovs))

""" Estimate channel coefficients"""
fovObjs = [FOV(fov, os.path.join(in_dir, fov), regex_3d, rnds, channels, normalize_max=normalize_ceiling, min_cutoff=min_intensity)#, imgfilter=smooth_method, smooth_param=sOrW) 
           for fov in chanCoef_fovs]

fovSubs = [fov.samplePixels(min_norm=min_dc_norm, sample_frac=chanCoef_frac) for fov in fovObjs]
fovjoin = xr.concat(fovSubs, dim='spatial')

coefs_df = estimateChannelCoefs(fovjoin, cb, n_iter=chanCoef_iter, min_norm=min_dc_norm)
# coefs_list = [ftreduce(lambda a, b: a*b, coefs_iters[:(i+1)]) for i in range(len(coefs_iters))]
# coefs_df = pd.DataFrame(coefs_list).T
# coefs_df.columns = ["iter{}".format(i) for i in range(len(coefs_df.columns))]

# write the channel coefs to a file with comments
with open(chanCoef_file, "w") as writer:
    writer.write("# samples fovs: {}\n".format(chanCoef_fovs))
    writer.write("# min_norm={}, sample_frac={}\n".format(min_dc_norm, chanCoef_frac))
    writer.write("# total pixels used: {}\n".format(fovjoin.shape[0]))
    coefs_df.to_csv(writer, sep="\t")        

"""Plotting the evolution of the coefficiens"""
plt.figure(figsize=(10, 4))
plt.vlines(np.arange(0, coefs_df.shape[0]+1) - 0.5, coefs_df.min().min(), coefs_df.max().max(), 
           linestyle='dashed', color='k', alpha=0.6)
for i in range(coefs_df.shape[1]):
    x = np.arange(0, coefs_df.shape[0]) + (i-coefs_df.shape[1]//2)/(coefs_df.shape[1]+1)
    plt.scatter(x, coefs_df.iloc[:, i], label='iter {}'.format(i), alpha=0.8)
plt.legend()
plt.xticks(np.arange(0, coefs_df.shape[0], 1), np.arange(0, coefs_df.shape[0], 1))
plt.xlabel('cycle-channel', fontsize=15)
plt.ylabel('coefficient', fontsize=15)
plt.title('Cycle-channel coefficients', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(chanCoef_plot, transparent=False, facecolor='white')


""" Run the decoding on all field of views"""
def dc_fov(name, indir, regex, rounds, chans, codebook, min_norm, alpha, chanCoefs, wthresh, outdir):
    fov = FOV(name, os.path.join(indir, name), regex, rounds, chans, 
              normalize_max=normalize_ceiling, min_cutoff=min_intensity, imgfilter=smooth_method, smooth_param=sOrW) 
    dcObj = normAndDeconv(fov.mip(), codebook=codebook, min_norm=min_norm, 
                          alpha=alpha, chanCoefs=chanCoefs)
    fov_spots = dcObj.createSpotTable(dcObj.ols_table, thresh_abs=wthresh, 
                                      flat_filter=None)
    fov_spots.to_csv(os.path.join(outdir, "{}_rawSpots{}.tsv".format(name, suffix)), sep="\t", float_format='%.3f')
    return 1

dcpartial = partial(dc_fov, indir=deepcopy(in_dir), regex=deepcopy(regex_3d), rounds=deepcopy(rnds), 
                    chans=deepcopy(channels), codebook=deepcopy(cb), min_norm=deepcopy(min_dc_norm), 
                    alpha=deepcopy(lasso_alpha), chanCoefs=deepcopy(coefs_df.values[:, -1]),
                    wthresh=deepcopy(weight_thresh), outdir=deepcopy(out_dir))
t1 = time.time()
Parallel(n_jobs=dc_npool, prefer='processes')(delayed(dcpartial)(name) for name in fov_names)
t2 = time.time()
print("It took {} seconds".format(t2 - t1))


# name = fov_names[0]
# fov = FOV(name, os.path.join(in_dir, name), regex_3d, rnds, channels, 
#           normalize_max=normalize_ceiling, min_cutoff=min_intensity) 
# amip = fov.mip()[:, :, 0:100, 0:100]
# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
# for i, ax in enumerate(axes.ravel()):
#     ax.imshow(amip[i].transpose().values)
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "test_raw.pdf"))


# name = fov_names[0]
# fov = FOV(name, os.path.join(in_dir, name), regex_3d, rnds, channels, 
#           normalize_max=normalize_ceiling, min_cutoff=min_intensity, imgfilter=smooth_method, smooth_param=sOrW) 
# amip = fov.mip()[:, :, 0:100, 0:100]
# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
# for i, ax in enumerate(axes.ravel()):
#     ax.imshow(amip[i].transpose().values)
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "test_median.pdf"))


# c = coefs_df.values[:, -1].reshape((8, 3))
# amip = fov.mip()[:, :, 0:100, 0:100] / c[..., None, None]
# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
# for i, ax in enumerate(axes.ravel()):
#     ax.imshow(amip[i].transpose().values)
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "test_median_norm.pdf"))