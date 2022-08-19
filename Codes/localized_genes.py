from copy import deepcopy
from skimage.color import hsv2rgb
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import label   
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.collections as pltCol
from matplotlib import pyplot as plt 
class DARTFISH_2D:
    def __init__(self, imgs, spots, topLeft = (0, 0), mask=None, name = None, condition=None, 
                 brightfieldChannel=None):
        """ imgs: a dict of dicts containing DART-FISH images. First layer keys are round names. 
            Second layer keys, if existing are channel names
            spots : a dataframe
            topLeft: tuple showing the global coordinates of the top left corner of the image
            mask: a Mask2D object or an array
            name: a string
            condition: a string
            bfCh: a stirng of the for chXX, showing the brightfield channel
        """
        self.imgs = imgs
        self.spots = spots
        self.topLeft = topLeft
        self.mask = mask
        self.name = name
        self.condition = condition
        self.bfCh = brightfieldChannel
        
    def getRoundRGB(self, rnd, hue_dict = {'ch00' : 1/3, 'ch02' : 1/2, 'ch03' : 0}, 
                 contrast_dict = {'ch00' : (10, 150), 'ch02' : (10, 150), 'ch03' : (10, 150)}, 
                    sat = 1):
        return get3ColorRGB(self.imgs[rnd], hue_dict, contrast_dict)
    
    def subsetCoords(self, p1, p2, isPixel = True):
        """ p1 and p2: 2-tuples containing the top-left and bottom-right corners
            isPixel: True if p1 and p2 are pixel coordinates, False if p1 and p2 are physical coordinates
        """
        if isPixel == False:
            raise NotImplementedError('Physical coordinates not implemented')
            
        subImgs = self._subsetAllImages(deepcopy(self.imgs), p1, p2)
        subSpots = DARTFISH_2D.subsetSpots(self.spots, p1, p2, shiftCoords = True)
        
        return DARTFISH_2D(subImgs, subSpots, (self.topLeft[0] + p1[0], self.topLeft[1] + p1[1]))
    
    def _subsetImage(self, img, p1, p2):   
        sImg = img.copy()
        sImg = sImg[p1[0]:p2[0], p1[1]:p2[1]]
        return sImg
    
    def _subsetAllImages(self, img_dict, p1, p2):
        for rnd in img_dict:
            if isinstance(img_dict[rnd], dict):
                self._subsetAllImages(img_dict[rnd], p1, p2)
            else:
                img_dict[rnd] = self._subsetImage(img_dict[rnd], p1, p2)
        return img_dict
    
    @staticmethod
    def subsetSpots(spots, p1, p2, shiftCoords = False, coords = ('xg', 'yg')):
        sspots = deepcopy(spots)
        sspots = sspots.loc[(sspots[coords[1]] >= p1[0]) & (sspots[coords[1]] <= p2[0])]
        sspots = sspots.loc[(spots[coords[0]] >= p1[1]) & (spots[coords[0]] <= p2[1])]
        if shiftCoords:
            sspots[coords[0]] = sspots[coords[0]] - p1[1]
            sspots[coords[1]] = sspots[coords[1]] - p1[0]
        return sspots
    
    def getGeneCounts(self):
        return self.spots.groupby('gene').size().sort_values(ascending = False)
    
    @staticmethod
    def segmentForeground(bfIm, entSize=10, entThresh=4, returnEntMap = False):
        """ bfIm: brightfield image
            entSize: integer. Radius of the disk on which the entropy is calculated.
            entThresh: threshold on the entropy 
            returnEntMap: return entropy map if True
            outputs a binary mask
        """
        ent_map = entropy(bfIm, disk(entSize))

        forg = binary_fill_holes(ent_map >= 4) #threshold_otsu(ent_map)
        labelIm = label(forg)
        assert (labelIm.max() > 0)
        bigCC = np.argmax(np.bincount(labelIm.flatten())[1:]) + 1
        forg = labelIm == bigCC
        
        if returnEntMap:
            return forg, ent_map
        else:
            return forg

    def setMask(self, mask):
        self.mask = mask
        
        
#     def plotOneGene(self, gene, ax = None, figsize = (10, 10), color = 'red', ptsize=2):
#         if ax is None:
#             fig, ax = plt.subplots(figsize = figsize)
        
    def plotGene(self, gene, color='red', backgImg=None, ax=None, figheight=15, coords=['xg', 'yg'], alpha=1, ptsize=1, backgCmap='gray'):
        spots = deepcopy(self.spots)
        spots = spots.loc[spots['gene'] == gene]
        
        if type(backgImg) == str:
            backgImg = self.imgs[backgImg]
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
        col1 = pltCol.PatchCollection(circ_patches, match_original=True)
        ax.add_collection(col1)
        return ax



import numpy as np, pandas as pd
from skimage.io import imread, imshow
from scipy.spatial import cKDTree

bcmag = "0.6"
spot_df = pd.read_csv("../3_Decoded/output_Starfish_bcmag{}/all_spots_filtered.tsv".format(bcmag), 
                      sep = '\t', index_col=0)
spot_df = spot_df.loc[spot_df['gene'] != 'Empty']

DF1 = DARTFISH_2D(None, spot_df)

def nnSpotFinder(allspots, gene, n_neighbors=5):
    spot_g = allspots.loc[allspots['gene'] == gene]
    spot_nn_g = cKDTree(spot_g[['xg', 'yg']])
    nearestDists, nearestInds = spot_nn_g.query(spot_g[['xg', 'yg']], k = range(2, 2 + n_neighbors))
    return nearestDists, spot_g.index[nearestInds]

n_neigh = 20

genes = DF1.getGeneCounts().index.to_numpy()
dist_stats = {}
for g in genes:
    print(g)
    dists, _ = nnSpotFinder(spot_df, gene=g, n_neighbors=n_neigh)
    cum_mean_perSpot = np.cumsum(dists, axis=1) / range(1, dists.shape[1] + 1)
    cum_mean_mean = cum_mean_perSpot.mean(axis=0)
    cum_mean_std = cum_mean_perSpot.std(axis=0)
    dist_stats[g] = {'dists_mean' : cum_mean_mean, 'dists_std' : cum_mean_std}
dist_stats = pd.DataFrame.from_dict(dist_stats, orient='index')


means_df = pd.DataFrame(dist_stats['dists_mean'].tolist(), index=dist_stats.index)
means_df.columns = ['mean_k{}'.format(i) for i in range(1, 1 + means_df.shape[1])]
std_df = pd.DataFrame(dist_stats['dists_std'].tolist(), index=dist_stats.index)
std_df.columns = ['std_k{}'.format(i) for i in range(1, 1 + std_df.shape[1])]
dist_stats = means_df.join(std_df)

stats_norm = dist_stats.loc[:, dist_stats.columns.str.startswith('mean')].multiply(np.sqrt(DF1.getGeneCounts()[dist_stats.index]), axis=0)

stats_norm = stats_norm.sort_values(by='mean_k3')
stat_plot = stats_norm.head(200)
fig, ax = plt.subplots(figsize = [20, 6], nrows=1)
ax.bar(stat_plot.index, stat_plot['mean_k3'])
ax.tick_params(axis='x', labelrotation = 90, labelsize = 7)
plt.tight_layout()
plt.savefig("../5_Analysis/localized_top200_k3.pdf")

stats_norm.to_csv("../5_Analysis/localized_scores.csv", sep="\t")
