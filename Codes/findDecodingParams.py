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

def gaussian(x, sigma):
    """ Return the normalized Gaussian with standard deviation sigma. """
    c = np.sqrt(2 * np.pi)
    return np.exp(-0.5 * (x / sigma)**2) / sigma / c


def estimateChannelCoefs(int_arr, codebook, alpha, ENargs, min_norm=0.3, n_iter=3):
    """ int_arr: Intensity data array, either with dims (RNDCH, spatial) or (RND, CHN, y, x)"""
    print("Channel estimation with data array size {}".format(int_arr.shape))
    cb_flat = codebook.stack(flatcode = (spd.RND, spd.CHN))
    coefs_list = [np.ones(cb_flat.shape[1])]

    for n in range(n_iter):
        intensities_ = deepcopy(int_arr)
        coefs = ftreduce(lambda a, b: a*b, coefs_list)
        t1= time.time()
        dcObj_ = spd.normAndDeconv(intensities_, cb, alpha, min_norm=min_norm, ENargs=ENargs, chanCoefs=coefs)
        t2 = time.time()
        print("Decoding took {} seconds".format(t2 - t1))

        """ Finding the coefficients with the new decoding results"""
        w_table = dcObj_.ols_table
        int_flat_ = dcObj_.ols_pixs.transpose()[:, (w_table.values > 0).sum(axis=0) == 1] # selecting pixels with only one dominant weight
        w_table = w_table[:, (w_table > 0).sum(axis=0) == 1] # selecting pixels with only one dominant weight
        print("pixels with pure weights : {}".format(w_table.shape))

        sumWeights = (cb_flat.values.T[:, None, :] * w_table.values.T).sum(axis=-1) # summing the weights in all pixels, in all cycles that the spots are "on"
        sumWeights = xr.DataArray(sumWeights, dims=('flatcode', 'pixels'), 
                                 coords={'rndch': ('flatcode', cb_flat.coords['flatcode'].values),
                                            spd.CHN : ('flatcode', cb_flat.coords[spd.CHN].values),
                                            spd.RND : ('flatcode', cb_flat.coords[spd.RND].values),
                                            'x' : ('pixels', w_table.coords['x'].values),
                                            'y' : ('pixels', w_table.coords['y'].values)})

        lr = LinearRegression(fit_intercept=False)
        newcoefs = []
        for i in range(sumWeights.shape[0]):
            x = sumWeights[i].values
            y = int_flat_[i].values
            y = y[(x > 0.1) & (x < 0.5)]
            x = x[(x > 0.1) & (x < 0.5)].reshape(-1, 1)
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

in_dir = params['proj_dir']
out_dir = params['dc_out']
plot_dir = os.path.join(out_dir,  "dcPlots") # where the decoding QC plots are saved

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

suffix = params['dc_suff']

rnds = params['dc_rounds']
channels = params['channel_order'] 
dc_npool = params['dc_npool']

pat3d = r"(?P<rndName>\S+)?_(?P<fov>FOV\d+)_(?P<ch>ch\d+)\S*.tif"
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
                    # 'alpha' : params['chan_coef_alpha'], # the regularization value for the elastic net model
                    'l1_ratio' : params['elasticnet_l1ratio'], # l1 ratio. 1 for a full lasso
                    'positive' : True, # forcing the model to search for positive coefficients
                    'selection' : params['elasticnet_selection'], # random vs. cyclic selection. 
                    'warm_start' : True, # may speed up computations a tiny bit
                    'fit_intercept' : False # our formulation doesn't need this
                    } # see sklearn.linear_model.ElasticNet for more details

if isinstance(chanCoef_fovs, int):
    chanCoef_fovs = list(np.random.choice(fov_names, chanCoef_fovs))

""" Estimate channel coefficients"""
fovSubs = []
for fov in chanCoef_fovs:
    temp_fov = FOV(fov, os.path.join(in_dir, fov), regex_3d, rnds, channels, normalize_max=normalize_ceiling, min_cutoff=min_intensity)
    temp_fov = temp_fov.samplePixels(min_norm=min_dc_norm, size=chanCoef_samps)
    fovSubs.append(temp_fov)
fovjoin = xr.concat(fovSubs, dim='spatial')    

coefs_df = estimateChannelCoefs(fovjoin, cb, alpha=params['chan_coef_alpha'], ENargs=elasticnet_params, n_iter=chanCoef_iter, min_norm=min_dc_norm)

# write the channel coefs to a file with comments
with open(chanCoef_file, "w") as writer:
    writer.write("# samples fovs: {}\n".format(chanCoef_fovs))
    writer.write("# min_norm={}, max sample_size per fov={}\n".format(min_dc_norm, chanCoef_samps))
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

""" Plotting some representing snippets of raw and normalized data, as well as the norms"""
k = cb.sum(dim=[spd.RND, spd.CHN]).values[0]
min_theo_norm = params['elasticnet_alpha'] * cb.shape[1] * cb.shape[2] / k # theoretical minimum of Lasso: lambda * N/k

for name in chanCoef_fovs:
    fov = FOV(name, os.path.join(in_dir, name), regex_3d, rnds, channels, 
              normalize_max=normalize_ceiling, min_cutoff=min_intensity, imgfilter=smooth_method, smooth_param=sOrW) 
    fov_mip = fov.mip()
    
    """ Plotting norms """
    ints_flat = fov_mip.stack(spatial=['z', 'y', 'x']).stack(RNDCH=[spd.RND, spd.CHN]).transpose('spatial', 'RNDCH')

    norms = np.linalg.norm(ints_flat.values, ord=2, axis=1)
    norms = norms[norms > 0.01]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(norms, bins=100)
    ax.set_xlim([0, 2])
    ax.set_yscale('log', nonpositive='clip')
    ax.set_title(fov.name + " norms")
    ax.set_xticks(np.arange(0, 2, 0.1), minor=True)
    ax.set_xticks(np.arange(0, 2, 0.2), ["{:0.1f}".format(x) for x in np.arange(0, 2, 0.2)], minor=False)
    y_max = ax.get_ylim()[1]
    ax.vlines(min_dc_norm, ymin=0, ymax=y_max, colors='orange', alpha=0.6, linestyle='dashed', label='norm threshold')
    ax.vlines(min_theo_norm, ymin=0, ymax=y_max, colors='red', alpha=0.6, linestyle='dashed', label='Lasso theoretical minimum')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "norm_hist_{}.pdf".format(fov.name)))

    """ Plotting raw intensities without normalization """
    amip = fov_mip[..., 400:600, 400:600]
    fheight = 9
    fwidth = fheight / 2 * (len(rnds)+1)//2
    fig, axes = plt.subplots(nrows=2, ncols=(len(rnds)+1)//2, figsize=(fwidth, fheight))
    for i in range(amip.shape[0]):
        axes.ravel()[i].imshow(amip[i].squeeze().transpose().values)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Example_{}_noNorm.pdf".format(name)))

    """ Plotting raw intensities WITH normalization """
    c = coefs_df.values[:, -1].reshape((len(rnds), 3))
    amip_norm = amip / c[..., None, None, None]
    fig, axes = plt.subplots(nrows=2, ncols=(len(rnds)+1)//2, figsize=(fwidth, fheight))
    for i in range(amip_norm.shape[0]):
        axes.ravel()[i].imshow(amip_norm[i].squeeze().transpose().values)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "Example_{}_norm.pdf".format(name)))
    plt.close('all')


coefs_df = pd.read_csv(chanCoef_file, sep="\t", comment="#", index_col=0)

""" Find the relationship between norm and best alpha for decoding"""
if not isinstance(params['elasticnet_alpha'], float):
    print("Estimating norm-alpha relationship")
    alpha_min, alpha_max = params['elasticnet_alpha']
    alphas = np.arange(alpha_min, alpha_max, 0.005)
    sig = 0.02

    t1 = time.time()

    dcpartial = partial(spd.normAndDeconv, codebook=cb, int_xarr=fovjoin, min_norm=min_dc_norm, ENargs=elasticnet_params, chanCoefs=coefs_df.values[:, -1], 
                                  elbow_thrs=elbow_thresholds)

    sObjs = Parallel(n_jobs=dc_npool, prefer='processes')(delayed(dcpartial)(alpha=al) for al in alphas)
    sObjs = {al : sObjs[i] for i, al in enumerate(alphas)}
    t2 = time.time()
    print("norm-alpha took {} seconds".format(t2-t1))

    def olsInLasso(ols_vec, lasso_inds):
        out = np.zeros(shape=lasso_inds.shape)
        out[lasso_inds] = ols_vec
        return out

    olssum_df = pd.DataFrame.from_dict({"ols_{}".format(al) : olsInLasso(sObjs[al].ols_table.sum('codes'), 
                                                                         sObjs[al].lasso_table.max('codes') > 0) 
                                                             for al in alphas})
    norms = np.linalg.norm(sObjs[alphas[0]].lasso_pixs, axis=1)

    test_norms = np.linspace(0, np.sqrt(8))

    bestalphas = [] 
    for al in test_norms:
        gaus_weights = gaussian(norms - al, sig)
        weighted_sum  = (olssum_df * gaus_weights.reshape(-1, 1)).sum(axis=0)
        bestalpha = weighted_sum.idxmax()
        bestalpha = float(bestalpha.split('_')[1])
        if len(bestalphas) > 0:
            bestalpha = np.maximum(np.max(bestalphas), bestalpha) # best alphas should be increasing with norm
        bestalphas.append(bestalpha)

    df = pd.DataFrame.from_dict({"norm" : test_norms, "alpha" : bestalphas})
    df.to_csv(os.path.join(out_dir, "norm_alpha_table{}.tsv".format(suffix)), sep="\t", index=False)

    plt.figure()
    plt.plot(test_norms, (bestalphas))
    plt.xlabel(r"norm", fontsize=12)
    plt.ylabel(r"Best $\alpha$ for deconvolution", fontsize=12)
    plt.savefig(os.path.join(plot_dir, "norm_alpha_curve.pdf"))

