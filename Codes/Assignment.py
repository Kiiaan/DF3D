import numpy as np
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.collections as col

class RolonyAssigner:
    """ A tool for assigning the rolonies to their closest nucleus, in both 2D and 3D. 
        Note: If the x-y-z coordinate of the image are not in the same order and the rolonies, 
        it will either throw an error or produce wrong results.
        nucleiImg: A 2D or 3D image/array with labeled nuclei. The whole volume of each 
            nucleus must be filled with a unique integer label.
        rolonyDf: A pandas dataframe containing the position of each rolony. 
        axes: A 1D list of the axes that must be used to calculate the distances.
            The elements in `axes` must be within the column names of rolonyDf. 
            If None, all columns in rolonyDf will be used for distance calculation.
        """
    def __init__(self, nucleiImg, rolonyDf, axes = None, flipCoords = False):
        """ The work flow is as follows:
            1. Make a binary image from the nucleiImg
            2. Find the object boundaries on this binary image
            3. Train a KD tree on these boundary positions
            4. Find the closest boundary pixel to each rolony
            5. Find the nucleus label of that boundary pixel. 
            6. If the rolony covers a nucleus, set the distance to 0
        """
        if axes is not None:
            rolonies = rolonyDf[axes].round().astype(int).values
        else:
            rolonies = rolonyDf.round().astype(int).values
        
        if flipCoords:
            print("Flipping the rolony coordinates")
            rolonies = np.flip(rolonies, axis = 1)
        
        nucBoundImg = find_boundaries(nucleiImg, mode = 'inner') # finding the inner boundaries
        boundPxls = np.transpose(np.nonzero(nucBoundImg)) # listing the boundary pixels
        kdtree = cKDTree(data = boundPxls) # training the kdtree
        self.nearestPxl_dist, nearestPxl_inds = kdtree.query(rolonies) # finding the nearest boundary pixels to rolonies
        
        """ iterating over the rolonies, writing the label of their nearest nucleus 
        and their distances to two vectors """
        self.nucLabels = nucleiImg[tuple(zip(*boundPxls[nearestPxl_inds]))]
        
        """ Finding the rolonies that directly cover a segmented nucleus"""
        in_inds = nucleiImg[tuple(zip(*rolonies))] > 0 
        self.nearestPxl_dist[in_inds] = 0
        
    def getResults(self):
        return self.nucLabels, self.nearestPxl_dist
    
def mask2rgb(mask, cmap):
    rgbmask = np.zeros(shape = (*mask.shape, 4))
    for i in range(mask.max() + 1):
        xx, yy = np.where(mask == i)
        rgbmask[xx, yy, :] = cmap(i)        
    return rgbmask

def plotRolonies2d(rolonyDf, nucLabels, coords = ['y', 'x'], label_name = 'nucleus_label', backgroudImg = None, ax = None, backgroundAlpha = 0.5):
    myCmap = np.random.rand(np.max(nucLabels) + 1, 4)
    myCmap[:, -1] = 1
    myCmap[0] = (0, 0, 0, 1)
    myCmap = ListedColormap(myCmap)
    # plt.style.use('dark_background')
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, figsize = (18, 11))
    
    boundaries = nucLabels.copy()
    boundaries[~find_boundaries(nucLabels)] = 0


    if not backgroudImg is None:
        ax.imshow(backgroudImg, alpha = backgroundAlpha, cmap = 'gray')
    
    # ax.imshow(mask2rgb(boundaries, myCmap), alpha = 0.7)#, vmin = 0, vmax = myCmap.N)
    # print(boundaries)
    ax.imshow(boundaries, cmap = myCmap, alpha=0.7)

    circ_patches = []
    for i, rol in rolonyDf.iterrows():
        circ = plt.Circle((rol[coords[0]], rol[coords[1]]), 2 * rol['radius'], 
                          linewidth = 0.7, fill = False, alpha = 0.8, 
                          color = myCmap(rol[label_name]))
        circ_patches.append(circ)
#         ax.add_patch(circ)
    # add the circles as a collection of patches (faster)
    col1 = col.PatchCollection(circ_patches, match_original=True)
    ax.add_collection(col1)
        # if (i %100) == 0:
        #     print(i)
    plt.tight_layout()
    plt.show()    

def plotRolonies2d2(rolonyDf, nucLabels, coords = ['y', 'x'], label_name = 'nucleus_label', backgroudImg = None, ax = None, backgroundAlpha = 0.5):
    myCmap = np.random.rand(np.max(nucLabels) + 1, 4)
    myCmap[:, -1] = 1
    myCmap[0] = (0, 0, 0, 1)
    myCmap = ListedColormap(myCmap)
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, figsize = (18, 11))
    
    boundaries = nucLabels.copy()
    boundaries[~find_boundaries(nucLabels)] = 0


    if not backgroudImg is None:
        ax.imshow(backgroudImg, alpha = backgroundAlpha, cmap = 'gray')
    
    # ax.imshow(mask2rgb(boundaries, myCmap), alpha = 0.7)#, vmin = 0, vmax = myCmap.N)
    # print(boundaries)
    ax.imshow(boundaries, cmap = myCmap, alpha=0.7)
    ax.scatter(rolonyDf[coords[0]], rolonyDf[coords[1]], s=0.3, alpha=0.8, color = list(map(lambda x: myCmap(x), rolonyDf[label_name])))
#     circ_patches = []
#     for i, rol in rolonyDf.iterrows():
#         circ = plt.Circle((rol[coords[0]], rol[coords[1]]), 2 * rol['radius'], 
#                           linewidth = 0.7, fill = False, alpha = 0.8, 
#                           color = myCmap(rol[label_name]))
#         circ_patches.append(circ)
# #         ax.add_patch(circ)
#     # add the circles as a collection of patches (faster)
#     col1 = col.PatchCollection(circ_patches, match_original=True)
#     ax.add_collection(col1)
        # if (i %100) == 0:
        #     print(i)
    plt.tight_layout()
    plt.show()    
