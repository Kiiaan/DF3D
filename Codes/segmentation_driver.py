import os
from os import path
from Segmentation import Segmentor2D
from Assignment import *
from skimage.io import imread
import numpy as np, pandas as pd
import multiprocessing
from functools import partial
from matplotlib.colors import ListedColormap
import yaml
import argparse
import threading

def mask2centroid(maskImg, ncore = 8):
    """ Finding centroids and area of segmented cells from a mask image """
    # ranges = np.split(np.arange(1, maskImg.max() + 1), ncore)
    ranges = np.split(np.arange(1, maskImg.max() + 1), np.linspace(1, maskImg.max() + 1, ncore+2).astype(int)[1:-1])

    pool = multiprocessing.Pool(ncore)
    f = partial(mask2centroid_parallel, mimg = maskImg)
    cent_arrs = list(pool.map(f, ranges))
    return np.concatenate(cent_arrs) 

def mask2centroid_parallel(rng, mimg):
    cent = []
    for i in rng:
        xs, ys = np.where(mimg == i)
        xc, yc = xs.mean().astype(int), ys.mean().astype(int)
        area = len(xs)
        cent.append((xc, yc, area))    
    return np.array(cent)
    
def cellmap_plot(cellInfos, bgImg, savepath, fwidth, fheight):
    print("Plotting cell map")
    fig = plt.figure(figsize = (fwidth, fheight))
    ax = fig.gca()
    ax.imshow(bgImg, cmap='gray')
    ax.scatter(cellInfos[:, 1], cellInfos[:, 0], s = 1, c='red')
    for i in range(cellInfos.shape[0]):
        ax.text(cellInfos[i, 1], cellInfos[i, 0], str(i), fontsize = 3, c = 'orange', alpha=0.8)
    fig.savefig(savepath,
                transparent = True, dpi = 400, bbox_inches='tight')
    print("Plotting cell map done")


parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

segm_type = params['segmentation_type']
flow_thresh = params['flow_threshold']
diam = params['seg_diam']

stitch_dir = params['stitch_dir']

if "pretrained_model" in params:
    pretrained_model = params['pretrained_model']
elif segm_type == 'nuc':
    pretrained_model = 'nuclei'
else:
    pretrained_model = 'cyto'


if 'nuc' in segm_type:
    nuc_path = os.path.join(stitch_dir, "{}_{}.tif".format(params['nuc_rnd'], params['nuc_ch']))
    nuc_img = imread(nuc_path)
    bgImg = nuc_img # background image

if 'cyto' in segm_type: 
    cyto_path = os.path.join(stitch_dir, "{}_{}.tif".format(params['cyto_rnd'], params['cyto_ch']))
    cyto_img = imread(cyto_path)
    bgImg = cyto_img # background image


saving_path = params['seg_dir']
if not path.exists(saving_path):
    os.makedirs(saving_path)

suff = params['seg_suf']

    
spot_file = os.path.join(params['dc_out'], 'all_spots_filtered.tsv')

# segmenting the nuclear image
if params['skip_seg']:
    mask = np.load(path.join(saving_path, 'segmentation_mask{}.npy'.format(suff)))
else:
    print('{} segmentation started.'.format(segm_type))
    
    segmentor = Segmentor2D(pretrained_model=pretrained_model, flow_threshold=flow_thresh)

    if segm_type == 'nuc':
        mask = segmentor.segment_singleChannel([nuc_img], diameters = diam, 
                             out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
    elif segm_type == 'cyto':
        mask = segmentor.segment_singleChannel([cyto_img], diameters = diam, 
                             out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
    elif segm_type == 'cyto+nuc' or segm_type == "nuc+cyto":
        mask = segmentor.segment_dualChannel([nuc_img], [cyto_img], diameters = diam, 
                             out_files = [path.join(saving_path, 'segmentation_mask{}.npy'.format(suff))])[0]
    print("Segmentation done.")
    

# Rolony assignment
spot_df = pd.read_csv(spot_file, index_col=0, sep = '\t')
assigner = RolonyAssigner(nucleiImg=mask, rolonyDf=spot_df, axes = ['yg', 'xg'])
labels, dists = assigner.getResults()

spot_df['cell_label'] = labels
spot_df['dist2cell'] = np.round(dists, 2)
spot_df = spot_df.sort_values('cell_label', ignore_index = True)
spot_df.to_csv(path.join(saving_path, 'spots_assigned{}.tsv'.format(suff)), sep = '\t', float_format='%.3f')


# finding the cells cell information: centroid and area
if not params['skip_seg']:
    cellInfos = mask2centroid(mask, ncore = params['centroid_npool'])
    centroid_df = pd.DataFrame({'cell_label' : np.arange(1, mask.max() + 1), 
                                'centroid_x' : cellInfos[:, 0], 'centroid_y' : cellInfos[:, 1],
                                'area' : cellInfos[:, 2]})
    centroid_df.to_csv(path.join(saving_path, 'cell_info{}.tsv'.format(suff)), sep = '\t', index = False)


# Making the cell by gene matrix
print('Making cell by gene matrix')
spot_df = spot_df.loc[spot_df['dist2cell'] <= params['max_rol2nuc_dist']] # filtering rolonies based on distance to cell
nuc_gene_df = spot_df[['cell_label', 'gene']].groupby(by = ['cell_label', 'gene']).size()
nuc_gene_df = nuc_gene_df.reset_index().pivot(index = 'cell_label', columns = 'gene').fillna(0).astype(int)
nuc_gene_df.columns = nuc_gene_df.columns.droplevel()
nuc_gene_df.to_csv(path.join(saving_path, 'cell-by-gene{}.tsv'.format(suff)), sep = '\t')

