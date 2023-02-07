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
from fieldOfView import FOV
import gc


def alphaAssign(x, x_norm, y_alpha):
    # For every value in vector x, find the index of the closest value in x_norm, and return y_alpha with that index
    return y_alpha[np.argmin(np.abs(x[:, None] - x_norm), axis=1)]

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

in_dir = params['reg_dir']
out_dir = params['dc_out']
plot_dir = os.path.join(out_dir,  "dcPlots") # where the decoding QC plots are saved

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

suffix = params['dc_suff']

is3D = params['deconv3d']
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
elbow_thresholds = params['elbow_thresholds'] # elbow detection thresholds (1 passes everything, 0 passes nothing)
min_maxWeight = params['min_maxWeight'] # thresh_abs in skimage's peak_local_max
weight_sigma = params['weight_smoothing_sigma'] # gaussian smoothing sigma on weight maps
min_weight = params['min_weight'] # min weight threshold smoothed maps
chanCoef_fovs = params['chan_coef_fovs'] # if list, name of fovs to use for channel coef estimation. If int, number of fovs to sample
chanCoef_samps = params['chan_coef_samples'] # fraction of pixels to sample from each fov to estimate channel coefficients
chanCoef_iter = params['chan_coef_niter']
chanCoef_file = os.path.join(out_dir, "channel_coefficients{}.tsv".format(suffix)) # params['chan_coef_file']
chanCoef_plot = os.path.join(plot_dir, "channel_coefficients{}.pdf".format(suffix)) # params['chan_coef_file']

elasticnet_params = {   
                    'l1_ratio' : params['elasticnet_l1ratio'], # l1 ratio. 1 for a full lasso
                    'positive' : True, # forcing the model to search for positive coefficients
                    'selection' : params['elasticnet_selection'], # random vs. cyclic selection. 
                    'warm_start' : True, # may speed up computations a tiny bit
                    'fit_intercept' : False # our formulation doesn't need this
                    # 'alpha' : params['elasticnet_alpha'], # the regularization value for the elastic net model
                    } # see sklearn.linear_model.ElasticNet for more details

coefs_df = pd.read_csv(chanCoef_file, sep="\t", comment="#", index_col=0)

if isinstance(params['elasticnet_alpha'], float):
    # no alpha fitting - constant alpha for all pixels
    alphaAssigner = params['elasticnet_alpha']
else:
    # read the relationship as a dataframe from file
    norm_alpha_df = pd.read_csv(os.path.join(out_dir, "norm_alpha_table{}.tsv".format(suffix)), sep="\t") 
    alphaAssigner = partial(alphaAssign, x_norm = norm_alpha_df['norm'].to_numpy(), y_alpha=norm_alpha_df['alpha'].to_numpy())


""" Run the decoding on all field of views"""
def dc_fov(name, indir, regex, rounds, chans, codebook, min_norm, alpha, ENargs, elbow_thrs, chanCoefs, min_peak, outdir, is3D, gaus_sigma, weight_thresh):
    if is3D:
        ints = FOV(name, os.path.join(indir, name), regex, rounds, chans, 
              normalize_max=normalize_ceiling, min_cutoff=min_intensity, imgfilter=smooth_method, smooth_param=sOrW).get_xr()
    else:
        ints = FOV(name, os.path.join(indir, name), regex, rounds, chans, 
              normalize_max=normalize_ceiling, min_cutoff=min_intensity, imgfilter=smooth_method, smooth_param=sOrW).mip()
    dcObj = spd.normAndDeconv(ints, codebook=codebook, alpha=alpha, min_norm=min_norm, 
                              ENargs=ENargs, chanCoefs=chanCoefs, elbow_thrs=elbow_thrs)
    fov_spots = dcObj.createSpotTable(dcObj.ols_table, thresh_abs=min_peak, gaus_sigma=gaus_sigma, weight_thresh=weight_thresh)
    fov_spots.to_csv(os.path.join(outdir, "{}_rawSpots{}.tsv".format(name, suffix)), sep="\t", float_format='%.3f')
    return 1


dcpartial = partial(dc_fov, indir=deepcopy(in_dir), regex=deepcopy(regex_3d), rounds=deepcopy(rnds), 
                    chans=deepcopy(channels), codebook=deepcopy(cb), min_norm=deepcopy(min_dc_norm), 
                    ENargs=deepcopy(elasticnet_params), chanCoefs=deepcopy(coefs_df.values[:, -1]),
                    min_peak=deepcopy(min_maxWeight), outdir=deepcopy(out_dir),
                    elbow_thrs=elbow_thresholds, is3D=is3D, gaus_sigma=weight_sigma, weight_thresh=min_weight,
                    alpha=alphaAssigner)

t1 = time.time()
Parallel(n_jobs=dc_npool, prefer='processes')(delayed(dcpartial)(name) for name in fov_names)
t2 = time.time()
print("It took {} seconds".format(t2 - t1))
