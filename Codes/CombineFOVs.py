""" This combines the spot tables from all FOVs. While combining, it removes duplicate spots
    from overlapping tiles and filters low quality spots.
"""
import os, re, numpy as np, pandas as pd
import functools
from copy import deepcopy
import yaml
import argparse
from skimage.filters import threshold_minimum
from scipy.spatial import cKDTree
# from utils import *

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import viridis
import matplotlib.patches as mpatches

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

def filterByEmptyRate():
    """ Filtering spots based on their area and max weight by calculating empty rate per spot. Steps:
        1. take a sample of spots from all field of views, keep their weight_max, area and gene. 
            Call these training spots
        2. Train a kDTree on the training spots with normalized weight_max and area as features. 
        3. For each non-empty spot, calculate the fraction of its k-nearest neighbors that are empty. 
            Call this value empty rate.
        4. (optional) If the histogram of empty rate is bimodal, find an empty rate threshold that 
            best separates spots with high empty rate from those with low empty rates
        5. Iterate over all FOVs and find the spot empty rates with the trained nearest neighbor
            model and apply the threshold to all spots
        5. Make suitable QC plots along the way for this process
    """
    """ 1. selecting a subsample of the data to train the nearest neighbor model"""
    print("Training the nearest neighbor model")
    spots_train = [pd.read_csv(file, index_col=0, sep="\t").assign(fov=re.search(fov_pat, file).group(0)).sample(frac=nn_sample_frac) for file in all_files]
    spots_train = pd.concat(spots_train, ignore_index=True)
    print('Training on {} spots'.format(spots_train.shape))
    
    """ 2. Training a KDTree to find local empty rate with 1st and 2nd distance2barcodes """
    X = spots_train[['weight_max', 'area']].to_numpy()
    X[:, 1] += np.random.normal(scale=0.3, size = X[:, 1].shape) # without this noise area becomes the dominant feature in nearest neighbor calculations
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std
    y = (spots_train['gene'] == 'Empty').to_numpy()
    nntree = cKDTree(X)

    """ 3. Calculating the fraction of empty barcodes within the k-nearest neighbor of each training spot"""
    _, er_train = nntree.query(X, k=n_neigh) # indices actually 
    er_train = y[er_train].mean(axis=1) # empty rate on training data

    """ 4. (optional) Calculating empty rate (er) distribution for training spots and find a threshold"""
    if empRateOrInfer == 'infer':
        empFrac = threshold_minimum(er_train[~y])
    else: 
        empFrac = empRateOrInfer

    """ 5 . (part 1) Plotts"""
    # the histogram of empty rates
    plt.figure()
    plt.hist(er_train[~y], bins=20)
    plt.vlines(empFrac, *plt.gca().get_ylim(), colors='red', label='threshold')
    plt.xlabel('Empty rate', fontsize=14)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_empty_rate_hist.pdf"))

    # scatter plots of weight and area with empty rates and rejection calls
    pass_patch = mpatches.Patch(color=viridis(0), label='pass filter')
    rjct_patch = mpatches.Patch(color=viridis(viridis.N), label='rejected')

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), sharex=True, sharey=True)
    ax = ax.ravel()
    ax[0].scatter(spots_train.loc[~y, 'weight_max'], spots_train.loc[~y, 'area'], s=0.1)
    ax[0].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[0].set_title("Non-empty spots")

    ax[1].scatter(spots_train.loc[y, 'weight_max'], spots_train.loc[y, 'area'], s=0.1)
    ax[1].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[1].set_ylabel("area", fontsize=12, fontweight='bold')
    ax[1].set_title("Empty spots")

    sc = ax[2].scatter(spots_train.loc[~y, 'weight_max'], spots_train.loc[~y, 'area'], c=er_train[~y], s=0.1)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax)
    ax[2].set_xlabel("weight_max", fontsize=12, fontweight='bold')
    ax[2].set_title("Empty rate")

    ax[3].scatter(spots_train.loc[~y, 'weight_max'], spots_train.loc[~y, 'area'], c=er_train[~y]>empFrac, s=0.1)
    ax[3].set_title("Passed and rejected spots")
    ax[3].legend(handles = [pass_patch, rjct_patch], prop={'size':14})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_empty_rate_scatter.png"), dpi=250)

    """ 4. Filtering spots in all FOVs and plotting results """
    passed_spots = []
    reject_spots = []
    for i, file in enumerate(all_files):
        if (i % 10) == 0:
            print('Cleaning {}'.format(re.search(fov_pat, file).group(0)))
        spots_fov = readSpots(file, FOVcoords)

        fov_X = spots_fov[['weight_max', 'area']].to_numpy()
        fov_X = (fov_X - X_mean) / X_std
        _, qInds = nntree.query(fov_X, k=n_neigh)
        spots_fov['EmptyRate'] = y[qInds].mean(axis=1)
        spots_pass = spots_fov.loc[spots_fov['EmptyRate'] <= empFrac]
        spots_rjct = spots_fov.loc[spots_fov['EmptyRate'] > empFrac]
        passed_spots.append(spots_pass)
        reject_spots.append(spots_rjct)
        
        """ 5. (part 2) scatter plots with rejection calls for all FOVs"""
        plt.figure(figsize=(6,5))

        plt.scatter(spots_pass['weight_max'], spots_pass['area'], s=0.1, c=np.array(viridis(0))[None])
        plt.scatter(spots_rjct['weight_max'], spots_rjct['area'], s=0.1, c=np.array(viridis(viridis.N))[None])
        plt.xlim([spots_train['weight_max'].min(), spots_train['weight_max'].max()])
        plt.ylim([spots_train['area'].min(), spots_train['area'].max()])
        plt.legend(handles = [pass_patch, rjct_patch], prop={'size':14})
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "distanceQC_{}.png".format(spots_fov['fov'].iloc[0])), dpi=150)
        plt.close()    

    passed_spots = pd.concat(passed_spots, ignore_index=True)
    reject_spots = pd.concat(reject_spots, ignore_index=True)
    return passed_spots, reject_spots

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

nn_sample_frac = params['distance_nn_frac'] # fraction of spots per FOV to use for training the nearest neighbor model
n_neigh = params['distance_nn_K']
empRateOrInfer = params['emptyRateThr'] # finding a distance that this fraction of decoded spots with smaller distances are empty
overlapRemovalRadius = params['overlapRemovalRadius'] # radius in pixels for removing overlaps

fov_pat = params['fov_pat'] # the regex showing specifying the tile names. 


plot_dir = os.path.join(decoding_dir,  "dcPlots") # where the decoding QC plots are saved

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

all_files = [os.path.join(decoding_dir, file)
             for file in os.listdir(decoding_dir)
             if re.search(fov_pat, file) and file.endswith('.tsv')]
all_files.sort(key = lambda x: int(re.search(fov_pat, x).group(1)))


""" Filtering spots per field of view based on their 1st and 2nd distance empty rate """
spots_passER, spots_rjctER = filterByEmptyRate()
spots_rjctER.reset_index(drop = True).to_csv(os.path.join(decoding_dir, 'qualityRejectedSpots.tsv'), sep = '\t')
spots_passER.reset_index(drop = True).to_csv(os.path.join(decoding_dir, 'qualityPassedSpots.tsv'), sep = '\t')
spots_passER = pd.read_csv(os.path.join(decoding_dir, 'qualityPassedSpots.tsv'), sep = '\t', index_col=0)

spots_overlapFree = removeOverlapRolonies(spots_passER, FOVmap, x_col='xg', y_col = 'yg', removeRadius=overlapRemovalRadius)
spots_overlapFree.to_csv(os.path.join(decoding_dir, "all_spots_filtered.tsv"), sep="\t")