""" This combines the spot tables from all FOVs. While combining, it removes duplicate spots
    from overlapping tiles and filters low quality spots.
"""
import os, re, numpy as np, pandas as pd
import functools
from copy import deepcopy
import yaml
import argparse
from skimage.filters import threshold_minimum
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
# from utils import *

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import viridis
import matplotlib.patches as mpatches
from time import time
import pickle

def removeOverlapRolonies(rolonyDf, fov_map, x_col = 'x', y_col = 'y', removeRadius = 5.5): #, n_pools=5):
    """ For each position, find those rolonies that are very close to other rolonies 
        in other positions and remove them.
        x_col and y_col are the names of the columns for x and y coordinates.
        removeRadius is in any unit that x_col and y_col are.
        fov_map is a dict showing the closest field of views to each FOV
    """
    geneList = rolonyDf.gene.unique()
    removePerGene = functools.partial(removeOverlapRolonies_gene, rolonyDf=rolonyDf, fov_map=fov_map, x_col=x_col, y_col=y_col, removeRadius=removeRadius)
    reducedRolonies = []
    for i, g in enumerate(geneList):
        print("{}. Working on gene {}".format(i, g))
        reducedRolonies.append(removePerGene(g))

    return pd.concat(reducedRolonies, ignore_index=True)

def removeOverlapRolonies_gene(gene, rolonyDf, fov_map, x_col = 'x', y_col = 'y', removeRadius = 5.5):
# def removeOverlapRolonies_gene(thisGene_rolonies, fov_map, x_col = 'x', y_col = 'y', removeRadius = 5.5):
    thisGene_rolonies = deepcopy(rolonyDf.loc[rolonyDf.gene == gene])
    for pos in sorted(thisGene_rolonies['fov'].unique()):
        thisPos = thisGene_rolonies.loc[thisGene_rolonies['fov'] == pos]
        otherPos = thisGene_rolonies.loc[thisGene_rolonies['fov'].isin(fov_map[pos])]
        if (len(thisPos) <= 0 ) or (len(otherPos) <= 0 ):
            continue
        nnFinder = cKDTree(thisPos[[x_col, y_col]])
        nearestDists, nearestInds = nnFinder.query(otherPos[[x_col, y_col]], distance_upper_bound = removeRadius)
        toRemoveFromThisPos_index = thisPos.index[nearestInds[nearestDists < np.inf]]
        thisGene_rolonies = thisGene_rolonies.drop(toRemoveFromThisPos_index)
    return(thisGene_rolonies)

def findFOVmap(coords_df):
    """ Take a dataframe of FOV coordinates, indexed by FOV name
        For each FOV, finds the 8 closest neighboring FOVs and returns them in a dict
    """
    coord_mat = coords_df.to_numpy()
    top8_dict = {}
    for i, fov in enumerate(coords_df.index):
        this_fov = coord_mat[i, ]
        x = coord_mat - this_fov
        dists = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
        top8 = np.argsort(dists)[1:9]
        top8 = coords_df.index[top8]
        top8_dict[fov] = top8
    return(top8_dict)

def trainClassifier(n_estimators=100, max_depth=4, class_weight='balanced', probability_threshold='infer', **kwargs):
    t1 = time()
    """ 1. selecting a subsample of the data to train the nearest neighbor model"""
    spots_train = [pd.read_csv(file, index_col=0, sep="\t").assign(fov=re.search(fov_pat, file).group(0)).sample(frac=nn_sample_frac) for file in all_files]
    spots_train = pd.concat(spots_train, ignore_index=True)
    

    """ 2. Prepping the data for the training """
    X = spots_train[['weight_max', 'area']].to_numpy()
    X[:, 1] += np.random.normal(scale=0.15, size = X[:, 1].shape) # A little randomization in area may help with the classification
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std
    y = (spots_train['gene'] == 'Empty').to_numpy()

    print('Training on an empty classifier on {} spots'.format(spots_train.shape[0]))
    rf1 = RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth, class_weight=class_weight, **kwargs)
    rf1.fit(X, y)
    t2 = time()
    print('Training the classifier took {} seconds.'.format(t2-t1))

    probs_train = rf1.predict_proba(X)[:, 1] # empty probability
    spots_train['EmptyProb'] = probs_train

    if probability_threshold == 'infer' or probability_threshold is None:
        prob_thresh = threshold_minimum(probs_train[~y])
    else:
        prob_thresh = probability_threshold
    
    """ Plots """
    # the histogram of empty probabilities
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 9))
    ax[0].hist(probs_train[~y], bins=50, label='Non-empty spots')
    ax[0].vlines(prob_thresh, ax[0].get_ylim()[0], ax[0].get_ylim()[1], colors='red', alpha=0.7, label='threshold')
    ax[0].legend(prop={'size': 12})
    ax[0].set_title("Empty probability histogram", fontsize=15, fontweight='bold')

    ax[1].hist(probs_train[y], bins=50, label='Empty spots', density=True)
    ax[1].vlines(prob_thresh, ax[1].get_ylim()[0], ax[1].get_ylim()[1], colors='red', alpha=0.7, label='threshold')
    ax[1].legend(prop={'size': 12})
    ax[1].set_xlabel('Empty probability', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_empty_prob_hist.pdf"))

    # scatter plots with empty probability as color
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), sharex=True, sharey=True)
    ax = ax.ravel()
    X_unscaled = X * X_std + X_mean
    ax[0].scatter(X_unscaled[~y, 0], X_unscaled[~y, 1], s=0.1)
    ax[0].set_ylim([0, np.percentile(X_unscaled[:, 1], 99.8) + 2])
    ax[0].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[0].set_title("Non-empty spots")

    # ax[1].scatter(spots_train.loc[y, 'weight_max'], spots_train.loc[y, 'area'], s=0.1)
    ax[1].scatter(X_unscaled[y, 0], X_unscaled[y, 1], s=0.1)
    ax[1].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[1].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[1].set_title("Empty spots")

    sc = ax[2].scatter(spots_train.loc[~y, 'weight_max'], spots_train.loc[~y, 'area'], c=probs_train[~y], s=0.1)
    ax[2].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[2].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[2].set_title("Empty probability of non-empty spots")
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax)

    sc=ax[3].scatter(spots_train.loc[y, 'weight_max'], spots_train.loc[y, 'area'], c=probs_train[y], s=0.1)
    ax[3].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[3].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[3].set_title("Empty probability of empty spots")
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_empty_rate_scatter.png"), dpi=250)


    # empty fraction vs empty probability cutoff
    C = np.linspace(0, 1)
    er = []
    for c in C:
        temp = spots_train.query("EmptyProb <= @c")
        er.append((temp['gene'] == 'Empty').sum() / temp.shape[0])
    plt.figure()
    plt.plot(C, er)
    plt.vlines(prob_thresh, plt.ylim()[0], plt.ylim()[1], colors='red', alpha=0.7, label='threshold')
    plt.xlabel("Empty probability cutoff", fontsize=12)
    plt.ylabel("Empty fraction passed filter", fontsize=12)
    plt.legend(prop={'size':14})
    plt.savefig(os.path.join(plot_dir, "probThreshold-v-emptyFraction.png"), dpi=250)
    
    return rf1, (X_mean, X_std), (ax[0].get_xlim(), ax[0].get_ylim()), prob_thresh

def readSpots(infile, coords_df=None, fov=None):
    """ Reads a dataframe for spots and adds global coordinates and other info 
        coords_df: if provided, is indexed by fov names
    """
    if fov is None:
        fov = re.search(fov_pat, infile).group(0)
    spts = pd.read_csv(infile, index_col=0, sep="\t").assign(fov=fov)
    spts['label'] = spts['fov'] + '_' + spts['label'].astype(str)

    if coords_df is None:
        return spts
    
    coords = coords_df.loc[fov]
    spts['xg'] = (coords_df.loc[fov, 'x'] + spts['x']).round(decimals=2)
    spts['yg'] = (coords_df.loc[fov, 'y'] + spts['y']).round(decimals=2)
    
    spts = spts[['xg', 'yg'] + [c for c in spts if not c in ['xg', 'yg']]]
    return spts

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

decoding_dir = params['dc_out'] # the main directory for decoding 

# if params['metadata_file'] is None:
#     metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
# else:
#     metadataFile = params['metadata_file']
    
# npix, vox, number_of_fovs = getMetaData(metadataFile)
# voxel_info = {"Y":vox['2'], "X":vox['1'], "Z":vox['3']}

FOVcoords = os.path.join(params['stitch_dir'], "registration_reference_coordinates.csv")
FOVcoords = pd.read_csv(FOVcoords).set_index('fov')
FOVcoords['x'] = FOVcoords['x'] - FOVcoords['x'].min() # making sure no pixels are in negative coordiates
FOVcoords['y'] = FOVcoords['y'] - FOVcoords['y'].min() # making sure no pixels are in negative coordiates
FOVmap = findFOVmap(FOVcoords) # dictionary to show the nearest tiles to each FOV

nn_sample_frac = params['distance_nn_frac'] # fraction of spots per FOV to use for training the classifier
empProbOrInfer = params['emptyProbabilityThr'] # finding a distance that this fraction of decoded spots with smaller distances are empty
n_estimators = params['randomForest_nEstimators'] # 100
max_depth = params['randomForest_maxDepth'] # 4

overlapRemovalRadius = params['overlapRemovalRadius'] # radius in pixels for removing overlaps

fov_pat = params['fov_pat'] # the regex showing specifying the tile names. 

plot_dir = os.path.join(decoding_dir,  "dcPlots") # where the decoding QC plots are saved

# legends for scatter plots
pass_patch = mpatches.Patch(color=viridis(0), label='pass filter')
rjct_patch = mpatches.Patch(color=viridis(viridis.N), label='rejected')

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

all_files = [os.path.join(decoding_dir, file)
             for file in os.listdir(decoding_dir)
             if re.search(fov_pat, file) and file.endswith('.tsv')]
all_files.sort(key = lambda x: int(re.search(fov_pat, x).group(1)))


clf, (X_mean, X_std), (xlim, ylim), prob_thresh = trainClassifier(n_estimators, max_depth, probability_threshold=empProbOrInfer)

""" Filtering spots per field of view based on the trained classifier and the learned threshold """
passed_spots = []
reject_spots = []
for i, file in enumerate(all_files):
    if (i % 10) == 0:
        print('Cleaning {}'.format(re.search(fov_pat, file).group(0)))
    spots_fov = readSpots(file, FOVcoords)

    fov_X = spots_fov[['weight_max', 'area']].to_numpy()
    fov_X = (fov_X - X_mean) / X_std
    spots_fov['EmptyProb'] = clf.predict_proba(fov_X)[:, 1]
    spots_pass = spots_fov.loc[spots_fov['EmptyProb'] <= prob_thresh]
    spots_rjct = spots_fov.loc[spots_fov['EmptyProb'] > prob_thresh]
    passed_spots.append(spots_pass)
    reject_spots.append(spots_rjct)
    
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharex=True, sharey=True)

    sc = axes[0].scatter(spots_fov['weight_max'], spots_fov['area'], s=0.1, c = spots_fov['EmptyProb'])
    axes[0].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("area", fontsize=12, fontweight='bold')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax)

    axes[1].scatter(spots_fov['weight_max'], spots_fov['area'], s=0.1, c=spots_fov['EmptyProb'] <= prob_thresh)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].legend(handles = [pass_patch, rjct_patch], prop={'size':14})
    axes[1].set_xlabel("weight_max", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "distanceQC_{}.png".format(spots_fov['fov'].iloc[0])), dpi=150)
    plt.close()    

passed_spots = pd.concat(passed_spots, ignore_index=True)
reject_spots = pd.concat(reject_spots, ignore_index=True)

reject_spots.reset_index(drop = True).to_csv(os.path.join(decoding_dir, 'qualityRejectedSpots.tsv'), sep = '\t')
passed_spots.reset_index(drop = True).to_csv(os.path.join(decoding_dir, 'qualityPassedSpots.tsv'), sep = '\t')

# Saving the classifier to file just in case
with open(os.path.join(decoding_dir, "QC_classifier.pickle"), "wb") as writer:
    pickle.dump((clf, X_mean, X_std, prob_thresh), writer)

passed_spots = pd.read_csv(os.path.join(decoding_dir, 'qualityPassedSpots.tsv'), sep = '\t', index_col=0)

spots_overlapFree = removeOverlapRolonies(passed_spots, FOVmap, x_col='xg', y_col = 'yg', removeRadius=overlapRemovalRadius)
spots_overlapFree.to_csv(os.path.join(decoding_dir, "all_spots_filtered.tsv"), sep="\t")