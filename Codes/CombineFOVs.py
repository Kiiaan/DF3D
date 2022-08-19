""" This combines the spot tables from all FOVs. While combining, it removes duplicate spots
    from overlapping tiles and filters spots based on their distance-to-barcode measure and 
    the empty-barcode rate. This whole process is repeated for all bcmags coming from StarFish.
    The output is written in the same directory that the spot tables are read from.
"""
import os, re, numpy as np, pandas as pd
import functools
from copy import deepcopy
import yaml
import argparse
from multiprocessing import get_context

from skimage.filters import threshold_minimum
from scipy.spatial import cKDTree
from utils import *

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
    # # with Pool(5) as P:
    # with get_context("spawn").Pool(5) as P:
    #     print('here')
    #     reducedRolonies = list(P.map(removePerGene, [deepcopy(rolonyDf.loc[rolonyDf.gene == gene]) for gene in geneList[:5]]))
    reducedRolonies = []
    for i, g in enumerate(geneList):
        print("{}. Working on gene {}".format(i, g))
        reducedRolonies.append(removePerGene(g))

    return pd.concat(reducedRolonies)

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

def filterByEmptyFraction():
    """ Filtering spots based on their 1st and 2nd distance to barcode and their local empty rate. Steps:
        1. take a sample of spots from all field of views, call this training spots
        2. using a KDTree, calculate the local empty rate for all training spots within 
            their K-nearest neighborhood, calculated using 1st and 2nd distance to barcode.
            K-nearest neighborhood are K decoded spots with most simialr 1st and 2nd distance
            to barcodes, some of which may be empty (invalid) barcodes
        3. find a local empty rate threshold that best separates spots with high empty rate from 
            those with low empty rates
        4. Iterate over all FOVs and find the spot empty rates with the trained nearest neighbor
            model and applying the threshold to all spots
        5. Make suitable QC plots along the way for this process
    """
    """ 1. selecting a subsample of the data to train the nearest neighbor model"""
    print("Training the nearest neighbor model")
    spots_train = [pd.read_csv(file, index_col=0).assign(fov=re.search(fov_pat, file).group(0)).sample(frac=nn_sample_frac) for file in all_files]
    spots_train = pd.concat(spots_train, ignore_index=True)
    # spots_train = spots_train.loc[(spots_train['area'] >= rolonyArea[0]) & (spots_train['area'] <= rolonyArea[1])]
    spots_train['gene'] = spots_train['target'].str.split('_').str[0]
    
    """ 2. Training a KDTree to find local empty rate with 1st and 2nd distance2barcodes """
    X = spots_train[['distance', 'second_distance']].to_numpy()
    y = (spots_train['gene'] == 'Empty').to_numpy()
    nntree = cKDTree(X)
    _, er_train = nntree.query(X, k=n_neigh) # indices actually 
    er_train = y[er_train].mean(axis=1) # empty rate on training data

    """ 3. Calculating empty rate (er) distribution for training spots and find a threshold"""
    if empFracOrInfer == 'infer':
        empFrac = threshold_minimum(er_train)
    else: 
        empFrac = empFracOrInfer

    """ 5 . (part 1) Plotts"""
    # the histogram of local empty rates
    plt.figure()
    plt.hist(er_train, bins=20)
    plt.vlines(empFrac, *plt.gca().get_ylim(), colors='red', label='threshold')
    plt.xlabel('Local empty rate', fontsize=14)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_emptyRate_hist.pdf"))

    # scatter plots of 1st and 2nd distances with empty rates and rejection calls
    pass_patch = mpatches.Patch(color=viridis(0), label='pass filter')
    rjct_patch = mpatches.Patch(color=viridis(viridis.N), label='rejected')

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharex=True, sharey=True)
    ax[0].scatter(X[:, 0], X[:, 1], c=y, s=0.1)
    ax[0].set_xlabel("1st distance", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("2nd distance", fontsize=12, fontweight='bold')
    ax[0].set_title("Empty spots")

    sc = ax[1].scatter(X[:, 0], X[:, 1], c=er_train, s=0.1)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax)
    ax[1].set_xlabel("1st distance", fontsize=12, fontweight='bold')
    ax[1].set_title("Local empty rate")

    ax[2].scatter(X[:, 0], X[:, 1], c=er_train>empFrac, s=0.1)
    ax[2].set_title("Passed and rejected spots")
    ax[2].legend(handles = [pass_patch, rjct_patch], prop={'size':14})
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_distance_plot.png"), dpi=250)

    """ 4. Filtering spots in all FOVs and plotting results """
    print("Applying the nearest neighbor filter to all FOVs")
    passed_spots = []
    reject_spots = []
    for i, file in enumerate(all_files):
        if (i % 10) == 0:
            print('Working on {}'.format(re.search(fov_pat, file).group(0)))
        spots_fov = readSpots(file)
        # spots_fov = spots_fov.loc[(spots_fov['area'] >= rolonyArea[0]) & (spots_fov['area'] <= rolonyArea[1])]
        spots_fov['gene'] = spots_fov['target'].str.split('_').str[0]
        fov_X = spots_fov[['distance', 'second_distance']].to_numpy()

        _, qInds = nntree.query(fov_X, k=n_neigh)
        spots_fov['localEmptyRate'] = y[qInds].mean(axis=1)
        spots_pass = spots_fov.loc[spots_fov['localEmptyRate'] <= empFrac]
        spots_rjct = spots_fov.loc[spots_fov['localEmptyRate'] > empFrac]
        passed_spots.append(spots_pass)
        reject_spots.append(spots_rjct)
        
        """ 5. (part 2) scatter plots with rejection calls for all FOVs"""
        plt.figure(figsize=(6,5))

        plt.scatter(spots_pass['distance'], spots_pass['second_distance'], s=0.1, c=np.array(viridis(0))[None])
        plt.scatter(spots_rjct['distance'], spots_rjct['second_distance'], s=0.1, c=np.array(viridis(viridis.N))[None])
        plt.legend(handles = [pass_patch, rjct_patch], prop={'size':14})
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "distanceQC_{}.png".format(spots_fov['fov'].iloc[0])), dpi=150)
        plt.close()    

    passed_spots = pd.concat(passed_spots, ignore_index=True)
    reject_spots = pd.concat(reject_spots, ignore_index=True)
    return passed_spots, reject_spots

def readSpots(infile):
    """ Reads a dataframe for spots and adds global coordinates and other info """
    spts = pd.read_csv(infile, index_col=0).assign(fov=re.search(fov_pat, infile).group(0))
    spts['xg'] = (round(spts['xc'] / voxel_info['X'])).astype(int)
    spts['yg'] = (round(spts['yc'] / voxel_info['Y'])).astype(int)
    spts['zg'] = (round(spts['zc'] / voxel_info['Z'])).astype(int)
    spts['spot_id'] = spts['fov'] + '_' + spts['spot_id'].astype(str)
    return spts
# def makeSpotTable(files_paths, emptyFractionCutoff, voxel_info, fov_map, removeRadius=5.5):
#     # Concatenating spots from all FOVs and converting the physical coordinates to pixels 
#     allspots = []
#     print("Reading spot for each FOV")
#     for file in all_files: 
#         print(file)
#         thisSpots = pd.read_csv(file, index_col = 0)
#         thisSpots['xg'] = (round(thisSpots['xc'] / voxel_info['X'])).astype(int)
#         thisSpots['yg'] = (round(thisSpots['yc'] / voxel_info['Y'])).astype(int)
#         thisSpots['zg'] = (round(thisSpots['zc'] / voxel_info['Z'])).astype(int)
#         thisSpots['fov'] = re.search(fov_pat, file).group()
#         thisSpots['spot_id'] = thisSpots['fov'] + '_' + thisSpots['spot_id'].astype(str)
#         allspots.append(thisSpots)

#     allspots = pd.concat(allspots, ignore_index=True)

#     allspots['gene'] = allspots['target'].str.extract(r"^(.+)_")

#     # removing empty spots of area 1 if they exist
#     allspots = allspots.loc[~((allspots['gene'] == 'Empty') & (allspots['area'] == allspots['area'].min()))]

#     allspots = allspots.sort_values('distance')

#     # Removing duplicate rolonies caused the overlapping regions of FOVs
#     print("Removing overlapping rolonies")
#     allspots_reduced = removeOverlapRolonies(allspots, fov_map, x_col='xg', y_col = 'yg', removeRadius=removeRadius)
    
#     # Keeping only spots with small distance to barcode so that `emptyFractionThresh` of spots are empty.
#     allspots_trimmed, allspots_highDist, allspots_reduced = filterByEmptyFraction(allspots_reduced, cutoff = emptyFractionCutoff)

#     return allspots_trimmed, allspots_reduced, allspots_highDist

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

decoding_dir = params['dc_out'] # the main directory for decoding 

bcmags = ["bcmag{}".format(params['bcmag'])]

if params['metadata_file'] is None:
    metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
else:
    metadataFile = params['metadata_file']
    
npix, vox, number_of_fovs = getMetaData(metadataFile)
voxel_info = {"Y":vox['2'], "X":vox['1'], "Z":vox['3']}

FOVcoords = os.path.join(params['stitch_dir'], "registration_reference_coordinates.csv")
FOVcoords = pd.read_csv(FOVcoords).set_index('fov')
FOVmap = findFOVmap(FOVcoords) # dictionary to show the nearest tiles to each FOV

nn_sample_frac = params['distance_nn_frac'] # fraction of spots per FOV to use for training the nearest neighbor model
n_neigh = params['distance_nn_K']
empFracOrInfer = params['emptyFractionThresh'] # finding a distance that this fraction of decoded spots with smaller distances are empty
overlapRemovalRadius = params['overlapRemovalRadius'] # radius in pixels for removing overlaps

fov_pat = params['fov_pat'] # the regex showing specifying the tile names. 

for bcmag in bcmags: 
    print("filtering barcode magnitude: {}".format(bcmag))
    savingpath = decoding_dir + "_" + bcmag
    plot_dir = os.path.join(savingpath,  "dcPlots") # where the decoding QC plots are saved
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    all_files = [os.path.join(savingpath, file)
                 for file in os.listdir(os.path.join(savingpath))
                 if re.search(fov_pat, file)]
    all_files.sort(key = lambda x: int(re.search(fov_pat, x).group(1)))

    """ Filtering spots per field of view based on their 1st and 2nd distance empty rate """
    spots_passER, spots_rjctER = filterByEmptyFraction()
    spots_rjctER.reset_index(drop = True).to_csv(os.path.join(savingpath, 'qualityRejectedSpots.tsv'), sep = '\t')
    spots_passER.reset_index(drop = True).to_csv(os.path.join(savingpath, 'qualityPassedSpots.tsv'), sep = '\t')
    # spots_passER = pd.read_csv(os.path.join(savingpath, 'qualityPassedSpots.tsv'), sep = '\t', index_col=0)

    spots_overlapFree = removeOverlapRolonies(spots_passER, FOVmap, x_col='xg', y_col = 'yg', removeRadius=overlapRemovalRadius)
    spots_overlapFree.to_csv(os.path.join(savingpath, "all_spots_filtered.tsv"), sep="\t")
    # filtered_spots, overlapFree_spots, removed_spots = makeSpotTable(all_files, emptyFractionThresh, VOXEL, FOVmap, removeRadius=overlapRemovalRadius)
    # filtered_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_spots_filtered_par.tsv'), sep = '\t')
    
    # overlapFree_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_spots_overlapFree_par.tsv'), sep = '\t')
    
    # removed_spots = removed_spots.loc[removed_spots['gene'] != 'Empty']
    # removed_spots.reset_index(drop = True).to_csv(os.path.join(savingpath, 'all_removed_spots_par.tsv'), sep = '\t')

    # fig, axes = plt.subplots(ncols = 2, figsize = (10, 6))
    # axes[0].plot(np.arange(0, overlapFree_spots.shape[0]), overlapFree_spots['cum_empty_frac'])
    # axes[0].set_xlabel("spot index")
    # axes[0].set_ylabel("empty fraction")

    # axes[1].plot(np.arange(0, overlapFree_spots.shape[0]), overlapFree_spots['distance'])
    # axes[1].set_xlabel("spot number")
    # axes[1].set_ylabel("barcode distance")
    # plt.tight_layout()
    # plt.savefig(os.path.join(savingpath, 'distance_emptyRate_plot_par.png'))
