import os, re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log 
from skimage.io import imread
import matplotlib.collections as col
import yaml
import argparse
from utils import getMetaData
import matplotlib.collections as col
from cellpose.plot import mask_overlay
from Assignment import plotRolonies2d2
from multiprocessing import Process

def normalize(im, ceil, amin=0):
    return (np.clip(im / ceil * 255, a_min = amin, a_max = 255)).astype(np.uint8)

def plotGene(spots, color='red', backgImg=None, ax=None, figheight=15, coords=['xg', 'yg'], alpha=1, ptsize=1, backgCmap='gray'):
    if ax is None:
        _, ax = plt.subplots(figsize = (figheight / backgImg.shape[0] * backgImg.shape[1], figheight))
    
    if not backgImg is None:
        ax.imshow(backgImg, cmap=backgCmap)
        
    circ_patches = []
    for i, (_, rol) in enumerate(spots.iterrows()):
        circ = plt.Circle((rol[coords[0]], rol[coords[1]]), ptsize, 
                          linewidth = 0.2, fill = True, alpha=alpha, color = color)
        circ_patches.append(circ)

    # add the circles as a collection of patches (faster)
    col1 = col.PatchCollection(circ_patches, match_original=True)
    ax.add_collection(col1)
    return ax

def maskOverlay_plot(mask, bgImg, savepath, fig, ax):
    print("Plotting the overlaid mask")
    ax.imshow(mask_overlay(bgImg, mask))
    fig.savefig(savepath, dpi = 500, bbox_inches='tight')
    print("Done plotting the overlaid mask")

def assignedRolonies_plot(fig, ax, spot_df, mask, bgImg, savepath, bgAlpha=0.8, coords=['xg', 'yg']):
    print("plotting assigned rolonies")
    plotRolonies2d2(spot_df, mask, coords = ['xg', 'yg'], label_name='cell_label', ax = ax, backgroudImg=bgImg, backgroundAlpha=bgAlpha)
    fig.savefig(savepath, transparent = False, dpi = 500, bbox_inches='tight')
    print("plotting assigned rolonies done")

def getHistHeight(histOut, x):
    n, bins, patches = histOut
    bar_n = np.where(x <= bins)[0][0] - 1
    h = n[bar_n]
    return h

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

spot_addr = os.path.join(params['seg_dir'], 'spots_assigned{}.tsv'.format(params['seg_suf']))
cellbygeneAddr = os.path.join(params['seg_dir'], 'cell-by-gene{}.tsv'.format(params['seg_suf']))

savingdir = params['qc_dir']

rolonyPlotDir = os.path.join(savingdir, "RolPlots")

if params['background_subtraction']:
    img_dir = params['background_subt_dir']
else:
    img_dir = params['proj_dir']


max_int = params['max_intensity'] # normalizing the image to this maximum
min_int = params['min_intensity']

anchor_pat = params['anc_rnd'] + "_{fov}_" + params['anc_ch'] + ".tif"
fovs = [file for file in os.listdir(img_dir) if file.startswith('FOV')]

if not os.path.isdir(savingdir):
    os.mkdir(savingdir)
if not os.path.isdir(rolonyPlotDir):
    os.mkdir(rolonyPlotDir)

spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']

# loading background image for whole tissue plots
if 'nuc' in params['segmentation_type']:
    nuc_path = os.path.join(params['stitch_dir'], "{}_{}.tif".format(params['nuc_rnd'], params['nuc_ch']))
    bgImg = imread(nuc_path)
    
if 'cyto' in params['segmentation_type']: 
    cyto_path = os.path.join(params['stitch_dir'], "{}_{}.tif".format(params['cyto_rnd'], params['cyto_ch']))
    bgImg = imread(cyto_path)

# the size for the spatial figures
fheight = 50
fwidth = int(fheight / bgImg.shape[0] * bgImg.shape[1])


# """ running blob detection on all FOVs """
# n_blobs = []
# n_rols = []
# all_blobs = []
# for fov in fovs: 
#     print(fov)
#     img_addr = os.path.join(img_dir, fov, anchor_pat.format(fov=fov))
#     anc_img = imread(img_addr)
#     this_blb = blob_log(normalize(anc_img, max_int, min_int), min_sigma=0.7, 
#                         max_sigma=2, num_sigma=6, overlap = 0.6, threshold = 0.3)
#     all_blobs.append(this_blb)
#     n_blobs.append(this_blb.shape[0])
#     n_rols.append(spot_df.loc[spot_df['fov'] == fov].shape[0])


# """ plotting decoded rolonies vs. anchor blobs """
# n_rols, n_blobs = np.array(n_rols), np.array(n_blobs)
# p = np.polyfit(n_blobs, y = n_rols, deg=1 )
# plt.figure()
# plt.plot([0, np.max(n_blobs)], [0, np.max(n_blobs)], c = 'green', alpha = 0.5, label = 'x=y line')
# plt.plot([0, np.max(n_blobs)], np.polyval(p, [0, np.max(n_blobs)]), c = 'orange', alpha = 0.5, label = 'fitted line')
# plt.scatter(n_blobs, n_rols, alpha =0.7)
# plt.ylabel("#decoded rolonies")
# plt.xlabel("#anchor blobs")
# plt.legend()
# plt.savefig(os.path.join(savingdir, "decoded_rolonies-v-anchor_blobs.pdf"))


# """ plotting decoding rate histogram """
# dec_rate = n_rols / n_blobs
# fig, ax = plt.subplots()
# ax.hist(dec_rate, bins = 20, alpha = 0.8)
# ax.vlines(np.median(dec_rate), *ax.get_ylim(), color = 'orange', label = 'median')
# ax.set_title("Decoding Rate Histogram")
# ax.set_xlabel("decoding rate")
# ax.set_ylabel("frequency")
# ax.legend()
# plt.savefig(os.path.join(savingdir, "decoding_rate_histogram.pdf"))


""" Plotting the rolonies, blobs and decoded rolonies"""
for i, fov in enumerate(fovs): 
    img_addr = os.path.join(img_dir, fov, anchor_pat.format(fov=fov))
    anc_img = imread(img_addr)
    if max_int == 'max':
        m_int = anc_img.max()
    else:
        m_int = max_int
    # this_blb = all_blobs[i]
    spot_df_fov = spot_df.loc[spot_df['fov'] == fov]
    fig, ax = plt.subplots(figsize = (20, 20))
    # this_blb = blob_log(normalize(anc_img, max_int, min_int), min_sigma=0.7, 
    #                     max_sigma=2, num_sigma=6, overlap = 0.6, threshold = 0.3)

    ax.imshow(normalize(anc_img, m_int, min_int), cmap = 'gray',alpha=0.9)

    # circ_patches = []
    # for x, y, r in this_blb:
    #     circ = plt.Circle((y, x), 3 * r, color = 'red', fill = None, alpha=0.8)
    #     circ_patches.append(circ)

    # add the circles as a collection of patches (faster)
    # col1 = col.PatchCollection(circ_patches, match_original=True)
    # ax.add_collection(col1)

    spot_df_fov = spot_df.loc[spot_df['fov'] == fov]
    circ_patches = []
    for i, row in spot_df_fov.iterrows():
        circ = plt.Circle((row['x'], row['y']), 3, color = 'lightgreen', fill = None, alpha=0.8)
        circ_patches.append(circ)

    # add the circles as a collection of patches (faster)
    col2 = col.PatchCollection(circ_patches, match_original=True)
    ax.add_collection(col2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(rolonyPlotDir, "{}.png".format(fov)))
    plt.close()


""" Plotting each individual gene"""
spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']
gene_counts = spot_df.groupby('gene').size().sort_values(ascending = False)
destdir = os.path.join(savingdir, "GenePlots")
if not os.path.isdir(destdir):
    os.makedirs(destdir)
    
genes = gene_counts.index.to_numpy()
for g in genes:
    print(g)
    spots_g = spot_df.loc[spot_df['gene'] == g]
    fig, ax = plt.subplots(figsize=(fwidth, fheight))
    ax.imshow(bgImg, cmap = "gray", vmax=400)
    plotGene(spots_g, ax=ax, ptsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(destdir, "{}_rolonies.pdf".format(g)))
    plt.close()

    

""" Rolony Stats """
cellbygene = pd.read_csv(cellbygeneAddr, sep = "\t", index_col=0)
n_cells = cellbygene.index[-1]
countpercell = cellbygene.sum(axis=1)
gene_per_nuc = (cellbygene > 0).sum(axis=1)
max_x_rols = np.floor(np.percentile(countpercell, 99))
max_x_genes = np.floor(np.percentile(gene_per_nuc, 99))

fig, ax = plt.subplots(ncols = 2, nrows=2, figsize = (12, 10))
ax = ax.flatten()

""" Rolonies per cell"""
O = ax[0].hist(countpercell, bins = 40, density=False, alpha = 0.8)
ax[0].set_title('#rolony count per cell histogram')
ax[0].set_yscale('log')
ax[0].vlines(np.mean(countpercell), ymin = 0, ymax = getHistHeight(O, np.mean(countpercell)), 
             color = 'red', label = "mean", linestyles='dashed')
ax[0].legend()


bins = np.arange(1, max_x_rols, 1)
count_cum = [np.sum(countpercell >= i) / n_cells for i in bins]
bins = np.append(0, bins) # have to append 0 and 1 since cells with no rolony are not in the count matrix
count_cum = np.append(1, count_cum)

ax[1].bar(bins, 100 * count_cum, width = 1, alpha = 0.8)
ax[1].set_yscale('log')
ax[1].set_xticks(np.arange(0, bins[-1], step = 10))
try:
    ax[1].vlines(x = bins[5], ymin = 0, ymax= 100 * count_cum[5], 
                 colors = 'red', linestyles='dashed', 
                 label = 'at least 5 rolonies')
except: 
    pass

try:
    ax[1].vlines(x = bins[20], ymin = 0, ymax= 100 * count_cum[20], 
                 colors = 'orange', linestyles='dashed', 
                 label = 'at least 20 rolonies')
except:
    pass
ax[1].set_title('Cumulative rolony count per nucleus')
ax[1].set_ylabel('% cells')
ax[1].set_xlabel('At least #rolony per cell')
ax[1].legend()

""" Genes per cell"""
bins = np.arange(1, max_x_genes, 1)
gene_cum = [np.sum(gene_per_nuc >= i) / n_cells for i in bins] 
bins = np.append(0, bins)
gene_cum = np.append(1, gene_cum)

O = ax[2].hist(gene_per_nuc, density=False, alpha = 0.8, bins=40)
ax[2].set_title('#Genes per cell distribution')
ax[2].set_yscale('log')
ax[2].vlines(np.mean(gene_per_nuc), ymin = 0, ymax = getHistHeight(O, np.mean(gene_per_nuc)), 
             color = 'red', label = "mean", linestyles='dashed')
ax[2].legend()

ax[3].bar(bins, 100 * gene_cum, width = 1, alpha = 0.8)
ax[3].set_yscale('log')
ax[3].set_xticks(np.arange(0, bins[-1], 5))
try:
    ax[3].vlines(x = bins[5], ymin = 0, ymax= 100 * gene_cum[5], 
                 colors = 'red', linestyles='dashed', 
                 label = 'at least 5 genes')
except:
    pass

try:
    ax[3].vlines(x = bins[20], ymin = 0, ymax= 100 * gene_cum[20], 
                 colors = 'orange', linestyles='dashed', 
                 label = 'at least 20 genes')
except:
    pass

ax[3].set_title('Cumulative #genes count per nucleus')
ax[3].set_ylabel('%cells')
ax[3].set_xlabel('at least #gene per cell')
ax[3].legend()

plt.tight_layout()
plt.savefig(os.path.join(savingdir, 'rolony_stats.png'), dpi=200)



""" cell size and radius histogram"""
cellInfo_df = pd.read_csv(os.path.join(params['seg_dir'], 'cell_info{}.tsv'.format(params['seg_suf'])), sep = "\t")

if params['metadata_file'] is None:
    metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
else:
    metadataFile = params['metadata_file']

_, vox, _ = getMetaData(metadataFile)

areaInPix = cellInfo_df['area']
diamInPix = np.sqrt(4 * areaInPix / np.pi)

areaInMicron = areaInPix * vox['2'] * vox['1']
diamInMicron = diamInPix * vox['1']

fig, axes = plt.subplots(ncols=2, nrows = 2, figsize = (15, 12))

axes = axes.ravel()
axes[0].hist(areaInPix)
axes[0].set_title("Cell Area (#pixels) Histogram")

axes[1].hist(areaInMicron)
axes[1].set_title("Cell Area (um^2) Histogram")

axes[2].hist(diamInPix)
axes[2].set_title("Cell Diameter (#pixels) Histogram")

axes[3].hist(diamInMicron)
axes[3].set_title("Cell Diameter (um) Histogram")

plt.tight_layout()
fig.savefig(os.path.join(savingdir, 'cell_size_stats.png'), dpi=200)


""" distance to cell histogram """
plt.figure(figsize = (8, 6))
plt.hist(spot_df['dist2cell'] * vox['1'], bins = 100)
plt.title("Distance to nearest cell histogram")
plt.xlabel("distance (um)")
plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(savingdir, 'dist2cell_hist.png'), dpi=200)


""" Empty rate per FOV """
spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
countPerFov = pd.crosstab(spot_df['gene'], spot_df['fov'])

plt.figure(figsize = (10 * len(countPerFov.columns) / 100, 6))
plt.bar(countPerFov.columns, countPerFov.loc['Empty'] / countPerFov.sum(axis=0))
_=plt.xticks(rotation=90, fontsize = 6)
plt.title("Empty rate per FOV")
plt.xlabel("FOV")
plt.ylabel("Empty rate")
plt.tight_layout()
plt.savefig(os.path.join(savingdir, 'EmptyRatePerFOV.png'), dpi=200)

""" Barcode digits per round """
spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']
spot_df['barcode'] = spot_df['target'].str.split("_").str[1]

barcode_mat = np.array([list(a) for a in spot_df['barcode']], dtype = int)
n_ch = barcode_mat.max()
bc_freq = np.array([np.sum(barcode_mat == i, axis=0) for i in range(1, n_ch + 1)])
bc_freq = bc_freq / barcode_mat.shape[0]

plt.figure()
plt.imshow(bc_freq, vmin = 0)
plt.colorbar()
plt.yticks(range(bc_freq.shape[0]), range(1, bc_freq.shape[0] + 1))
plt.xticks(range(bc_freq.shape[1]), ["dc{}".format(i) for i in range(bc_freq.shape[1])])
plt.title("Barcode Digit Frequency")
plt.savefig(os.path.join(savingdir, 'barcode_digit_frequency.png'))


""" Plotting the gene count barplot """
spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']
gene_counts = spot_df.groupby('gene').size().sort_values(ascending = False)

fig, ax = plt.subplots(figsize = [15, 6], nrows=1)
gene_counts = spot_df.groupby('gene').size().sort_values(ascending = False)
ax.bar(gene_counts.index[:100], gene_counts[:100])
ax.set_yscale('log')
ax.tick_params(axis='x', labelrotation = 90, labelsize = 9)
ax.set_ylabel('#rolony')
ax.set_ylim([1, np.max(gene_counts)*1.4])
ax.set_title('Top 100 Genes')
plt.savefig(os.path.join(savingdir, 'top_gene_counts.pdf'))


""" Plotting all decoded rolonies """
spot_df = pd.read_csv(spot_addr, sep = "\t", index_col = 0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']

fig, ax = plt.subplots(figsize = (fwidth, fheight))
ax.imshow(bgImg, cmap = 'gray')

patches = []
coords = ['xg', 'yg']

ax.scatter(spot_df[coords[0]], spot_df[coords[1]], s=.3, alpha=0.5, c = '#FFD64B')
plt.tight_layout()
plt.show()        
fig.savefig(os.path.join(savingdir, 'all_decoded_rolonies.pdf'), dpi = 250)

# """ plot segmentation mask """
mask = np.load(os.path.join(params['seg_dir'], 'segmentation_mask{}.npy'.format(params['seg_suf'])))
fig1, ax1 = plt.subplots(figsize=(fwidth, fheight))
maskO_thr = Process(target=maskOverlay_plot, kwargs={'mask':mask , 'bgImg' : bgImg, 
                                                            'savepath' : os.path.join(savingdir, 'mask{}.png'.format(params['seg_suf'])), 
                                                            'fig' : fig1, 'ax' : ax1})
maskO_thr.start()


""" plotting assigned rolonies"""
fig2, ax2 = plt.subplots(figsize = (fwidth, fheight))
assRol_thr = Process(target=assignedRolonies_plot, 
                                kwargs = {'spot_df' : spot_df, 
                                'mask' : mask, 'bgImg' : bgImg, 
                                'fig':fig2, 'ax':ax2,
                                'savepath' : os.path.join(savingdir, 'assigned_rolonies{}.png'.format(params['seg_suf']))})
assRol_thr.start()

""" plotting the cells with their label"""
cellInfos = pd.read_csv(os.path.join(params['seg_dir'], 'cell_info{}.tsv'.format(params['seg_suf'])), sep="\t").to_numpy()[:, 1:]
cellm_thr = Process(target = cellmap_plot, 
                                kwargs = {'cellInfos' : cellInfos, 'bgImg' : bgImg, 
                                'savepath' : os.path.join(savingdir, 'cell_map{}.png'.format(params['seg_suf'])), 
                                'fwidth' : fwidth, 'fheight' : fheight})
cellm_thr.start()

maskO_thr.join()
assRol_thr.join()
cellm_thr.join()
