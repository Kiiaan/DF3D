import numpy as np, xarray as xr, pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression
from copy import deepcopy
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball
from functools import reduce, partial
from scipy.ndimage import gaussian_filter
# from joblib import Parallel, delayed
# from multiprocessing import Pool
import gc
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RND, CHN = 'rnd', 'ch'

class SparseDecoder():
    def __init__(self, intensities, codebook, alpha, size=None, verbose=False, ENargs={}, min_norm=0.5):
        """ intensities: xr.DataArray. dims either
                1) (RND, CHN, y, x)
                2) (spatial, RNDCH). ُُThis shape can be obtained by using DataArray.stack()
            alpha: int or a vectorized function. If int, same value to be used for all pixels. 
                    If a function, pixel norms will be used as input to generate the values for alpha
            ENargs: dictionary for sklearn.linear_model.ElasticNet parameters
        """
        self.ints = intensities
        self.cb = codebook
        self.EN = self._setupElasticNet(ENargs)
        self.verbose = verbose
        self.norms = None # will be set during prepping pixels
        self.size = size # size of the field. Is necessary only if result image is going to be created
        self.lasso_pixs = None # will be set during prepping pixels
        self.min_norm = min_norm
        self.alphaGenerator = alpha
        self._prepTrainingPixels(self.min_norm)

    def _prepTrainingPixels(self, min_norm=0.5):
        """ Finding pixels that the lasso model will be trained on. The output, self.lasso_pixs is of the form ('spatial', 'RNDCH')"""
        if self.ints.dims == (RND, CHN, 'y', 'x'):
            pixel_intensities = self.ints.stack(spatial=['y', 'x']).stack(RNDCH=[RND, CHN]).transpose('spatial', 'RNDCH')
            self.size = (self.ints['x'].shape[0], self.ints['y'].shape[0], 1)
        if self.ints.dims == (RND, CHN, 'z', 'y', 'x'):
            pixel_intensities = self.ints.stack(spatial=['z', 'y', 'x']).stack(RNDCH=[RND, CHN]).transpose('spatial', 'RNDCH')
            self.size = (self.ints['x'].shape[0], self.ints['y'].shape[0], self.ints['z'].shape[0])
        
        elif self.ints.dims == ('spatial', 'RNDCH'):
            pixel_intensities = deepcopy(self.ints)
        else: 
            raise ValueError("Incorrect dimensions for self.ints: {}".format(self.ints.dims))
        norms = np.linalg.norm(pixel_intensities.values, ord=2, axis=1)
        self.lasso_pixs = pixel_intensities[norms >= min_norm]

        # preparing the alphas
        self.alphas = np.zeros(shape=self.lasso_pixs.shape[0])
        if callable(self.alphaGenerator):
            self.alphas = self.alphaGenerator(norms[norms >= min_norm])
        elif isinstance(self.alphaGenerator, float):
            self.alphas += self.alphaGenerator
        else:
            raise ValueError('SparseDecoder.alphaGenerator should either be a float or callable.')

        del self.ints # we no longer need all the intensities
        gc.collect()

    def applyLasso(self):
        if self.lasso_pixs is None:
            raise ValueError("training pixels aren't set yet!")
        
        cb_flat = self.cb.stack(flatcode = (RND, CHN))
        cb_fit = cb_flat.values.T

        if self.lasso_pixs.shape[0] == 0:   # no pixels with enough fluorescence
            if self.verbose:
                print("No pixels with enough fluorescence. Skipping deconvolution")
            self.lasso_table = None
            return

        """ Fitting the Lasso model"""
        if self.verbose:
            print("Starting the lasso fit. Data shape: {}".format(self.lasso_pixs.shape))

        # weights=np.array(Parallel(n_jobs=2, prefer='processes')(delayed(fit_partial)(row) for row in self.lasso_pixs))
        # p = Pool(processes = 4)
        # weights = np.array(list(p.map(fit_partial, self.lasso_pixs)))
        # print(self.lasso_pixs.shape)
        # from time import time
        # t1 = time()
        # print("doing nothing")
        # while (time() - t1) < 100:
        #     continue

        fit_partial = partial(self.fitEN, model=self.EN, cdbook=cb_fit, calculate_r2=False)
        weights = np.array(list(map(fit_partial, self.lasso_pixs.values, self.alphas)))

        self.lasso_table = xr.DataArray(weights.T,
                                        coords={'x':('pixels', self.lasso_pixs.coords['x'].values),
                                              'y':('pixels', self.lasso_pixs.coords['y'].values),
                                              'z':('pixels', self.lasso_pixs.coords['z'].values),
                                              'target':('codes', cb_flat.coords['target'].values),
                                              'gene' : ('codes', list(map(lambda x: x.split('_')[0], cb_flat.coords['target'].values)))}, 
                                        dims=['codes', 'pixels'])
        if self.verbose:
            print("Done fitting lasso")

    def applyOLS(self, elbow_thrs=[0.1, 0.1]):
        if self.lasso_table is None:
            self.ols_table = None
            return
        if self.lasso_table.values.max(axis=0).sum() == 0: # no pixels having weights
            self.ols_table = deepcopy(self.lasso_table)
            return
        
        ols = LinearRegression(fit_intercept=False) # positive=True
        cb_flat = self.cb.stack(flatcode = (RND, CHN)) # flatten codebook  
        w_table = deepcopy(self.lasso_table)
        self.ols_pixs = self.lasso_pixs[(w_table.max(axis=0).values > 0)] # removing pixels with 0 weight
        w_table = w_table.where(w_table.max(axis=0) > 0, drop=True) # removing pixels with 0 weight
        w_numpy = self.elbowFilter(w_table, abs_thr=elbow_thrs).values # select barcodes with lasso. weight will be updated with ols weights
        for j in range(w_table.shape[1]): # iterate over every pixel 
            selinds = np.nonzero(w_numpy[:, j])[0] # non-zero weight indices (after filtering)
            if selinds.shape[0] == 0:
                continue
            selbc = cb_flat[selinds].values.T # the barcodes with non-zero weights
            wj = self.myfit(self.ols_pixs[j].values, ols, selbc, calculate_r2=False) # the ols fit
            w_numpy[selinds, j] = wj # replacing the lasso weights with ols weights
        self.ols_table = xr.DataArray(w_numpy, dims = w_table.dims,
                                      coords = w_table.coords)
        
    def make4DWeightMap(self, w_table, size=None, bc_list=None, gene_list=None):
        """ Makes a 4 dimensional image with dimensions: (codes, z, y, x) populated with barcode weights
            w_table: a DataArray with dims (codes, pixels) of barcodes weights for each pixel
            bc_list or gene_list: a list or "all". Only one can be set
        """
        if self.size is None:
            raise ValueError("Field size is not set and w4d cannot be constructed")
        else:
            n_x = self.size[0]
            n_y = self.size[1]
            n_z = self.size[2]
        
        if (bc_list is None) and (gene_list is None):
            raise ValueError("Both barcode list and gene list are None. One needs to be set.")

        if (bc_list is not None) and (gene_list is not None):
            raise ValueError("Both barcode list and gene list cannot be set at once.")

        if bc_list == 'all':
            bcs = self.cb['target'].values
        elif not bc_list is None:
            bcs = bc_list

        """ If gene list is given, we have to convert it to a barcode list first"""
        if gene_list == "all":
            bcs = self.cb['target'].values
        elif not gene_list is None: # this keeps the order of the genes
            bcs = reduce(lambda x, y: x+y, [list(self.cb['target'][self.cb['gene'] == g].values) for g in gene_list])
        genes = [bc.split('_')[0] for bc in bcs]

        bc_inds = [np.where(w_table['target'].values == bc)[0][0] for bc in bcs]
        wbc_table = w_table[bc_inds]
        wbc_table = wbc_table[:, wbc_table.sum(dim='codes') > 0]

        # declare a zero image and populate it with the deconvolved weights
        w4d = xr.DataArray(np.zeros((len(bcs), n_z, n_y, n_x)), 
                                dims=['codes', 'z', 'y', 'x'],
                                coords = {'target' : ('codes', bcs),
                                          'gene' : ('codes', genes),
                                          'x': range(n_x), 'y': range(n_y), 
                                          'z': range(n_z) }) 
        
        w4d[wbc_table['codes'], wbc_table['z'], wbc_table['y'], wbc_table['x']] = wbc_table
        return w4d
    
    @staticmethod
    def myfit(int_row, model, cdbook, calculate_r2=False):
        """ int_row: array row of intensities
            cdbook: array of the codebook with flattened (onehot) barcodes
            model: a sklearn model
        """
        model.fit(cdbook, int_row)
        if calculate_r2:
            coefs = model.coef_
            r2 = model.score(cdbook, int_row)
            return np.append(coefs, r2)
        return list(model.coef_) # don't know why, but doesn't work without the list

    def fitEN(self, int_row, alpha, model, cdbook, calculate_r2=False):
        """ int_row: array row of intensities
            cdbook: array of the codebook with flattened (onehot) barcodes
            model: a sklearn model
        """
        model.alpha = alpha
        model.fit(cdbook, int_row)
        if calculate_r2:
            coefs = model.coef_
            r2 = model.score(cdbook, int_row)
            return np.append(coefs, r2)
        return list(model.coef_) # don't know why, but doesn't work without the list


    def _setupElasticNet(self, ENargs):
        # alpha = ENargs['alpha'] if ("alpha" in ENargs) else 0.02 # may be overwritten
        l1_ratio = ENargs['l1_ratio'] if ("l1_ratio" in ENargs) else 0.99
        positive = ENargs['positive'] if ("positive" in ENargs) else True
        selection = ENargs['selection'] if ("selection" in ENargs) else "random"
        warm_start = ENargs['warm_start'] if ("warm_start" in ENargs) else True
        fit_intercept = ENargs['fit_intercept'] if ("fit_intercept" in ENargs) else False
        return ElasticNet(l1_ratio=l1_ratio, positive=positive, fit_intercept=fit_intercept,
                            selection=selection, warm_start=warm_start)
            
    def getResultImage(self, method='lasso'):
        if method == 'lasso':
            return self.createResultImage(self.lasso_table)
        elif method == 'ols':
            return self.createResultImage(self.ols_table)
        else:
            raise ValueError('No such method {} implemented'.format(method))  

    def createSpotTable(self, w_table, 
                        flat_filter_kwargs={}, #volume_filter_kwargs={}, 
                        thresh_abs=0.2, peak_footprint=2, peak_mindistnce=1,
                        projectWeights = True, gaus_sigma=0, weight_thresh=0.01):
        """ Filtering the weights, applying watershed segmentation to each barcode, 
                and summarizing the segmented regions. 
            thresh_abs, peak_footprint and peak_mindistnce are inputs to self.segmentWeights
                (technically inputs to peak_local_max)
            projectWeights: if doing 3D decoding, all weights are maximum projected on z-axis
            gaus_sigma: sigma of the filter
        """
        if self.verbose:
            print("Performing flat filtering on weights")
        
        # iterating over barcodes and extract spots from each
        spots = []
        for bc in self.cb['target'].values:
            if self.verbose:
                print("Performing watershed segmentation on every barcode")
            wmap = self.make4DWeightMap(w_table, bc_list=[bc]).squeeze()
            
            if 'z' in wmap.dims:
                filt = partial(gaussian_filter, sigma=[0, gaus_sigma, gaus_sigma])
            else:
                filt = partial(gaussian_filter, sigma=[gaus_sigma, gaus_sigma])
            wmap.values = filt(wmap.values)
            wmap = wmap.where(wmap >= weight_thresh, other=0)

            if projectWeights and ('z') in wmap.dims:
                wmap = wmap.max('z') # maximum weight projection in z
            
            spots_g = SparseDecoder.segmentWeights(wmap.values, thresh_abs=thresh_abs, peak_footprint=peak_footprint, peak_mindistnce=peak_mindistnce)
            if spots_g.shape[0] == 0:
                continue
            spots_g['target'] = bc
            spots_g['gene'] = spots_g['target'].str.split('_').str[0]
            spots_g['label'] = bc + "_" + spots_g['label'].astype(str)
            spots.append(spots_g)
        
        if len(spots) > 0:
            self.dc_spots = pd.concat(spots, ignore_index=True)
        else:
            self.dc_spots = pd.DataFrame()

        return deepcopy(self.dc_spots)
        
    @staticmethod
    def topNFilter(weights, n=2):
        """ For every spatial location, keeps the top n barcode weights, fills the rest with 0
            weights: xarray with dimensions bc, z, y, x"""
        w_ntop = np.sort(weights.values, axis=0)[-n] # second to highest weight
        return weights.where(weights >= w_ntop[None], 0) #where the filtering happens
    
    @staticmethod
    def elbowFilter(wtable, n=2, abs_thr = [0.1, 0.1], returnThresh=False):
        """Keeps the top `n` weights if they are dominant. Procedure:
            For every pixel in w_table (row), sorts the weights in descending order and normalizes them to 
            have max of 1, finds the index at which there is a large dip in weights. If the dip happens in 
            the first `n` indices, the weights before the dip are kept, otherwise all weights are set to 0.
            Variables:
            w_table: DataArray with barcodes in rows, pixels in columns, populated by weights
            n: upper bound on where the dip in weights occurs
            abs_thr: a list of length n. A dip is called at position i if the normalize weight at 
                        position i+1 is smaller than abs_thr[i]
        """
        if type(abs_thr) is not list:
            abs_thr = n * [abs_thr]
        elif len(abs_thr) != n:
            raise ValueError('n and length of abs_thr not equal')
    
        warr = wtable.values

        warr_sort = np.array([col[np.argsort(-col)] for col in warr.T]).T # sorted weights descending order
        warr_norm = warr_sort / warr_sort.max(axis=0) # normalized to have max weight of 1

        w_thr = np.zeros(wtable.shape[1]) # vector to hold threshold values for pixels
        w_thr[:] = np.inf # anything that's not set below will never pass threshold

        for i in range(n): # iterate over the first n positions
            if i == 0:
                w_thr[warr_norm[i+1, :] <= abs_thr[i]] = warr_sort[i, warr_norm[i+1, :] <= abs_thr[i]] # update threshold values if dips are identified
            if i > 0:
                inds2upd = (warr_norm[i+1, :] <= abs_thr[i]) & (warr_norm[i, :] > abs_thr[i]) # dips should not carry over to next positions
                w_thr[inds2upd] = warr_sort[i, inds2upd] # update threshold values if dips are identified
       
        
        out = wtable.where(wtable >= w_thr, other=0)

        if returnThresh: 
            return out, w_thr
        else: 
            return out

    @staticmethod
    def elbowFilter_old(wtable, n=2, diff_thr = [0.5, 0.4], returnThresh=False):
        """Keeps the top `n` weights if they are dominant. Procedure:
            For every pixel in w_table (row), sorts the weights in descending order and normalizes them to 
            have max of 1, finds the index at which there is a large dip in weights. If the dip happens in 
            the first `n` indices, the weights before the dip are kept, otherwise all weights are set to 0.
            Variables:
            w_table: DataArray with barcodes in rows, pixels in columns, populated by weights
            n: upper bound on where the dip in weights occurs
            diff_thr: a list of length n, or a float. A dip is called at position i if the
                        1st order difference surpasses diff_thr[i-1]
        """
        if type(diff_thr) is not list:
            diff_thr = n * [diff_thr]
        elif len(diff_thr) != n:
            raise ValueError('n and length of diff_thr not equal')
    
        warr = wtable.values

        warr_sort = np.array([col[np.argsort(-col)] for col in warr.T]).T # sorted weights descending order
        warr_norm = warr_sort / warr_sort.max(axis=0) # normalized to have max weight of 1

        wdiff = -np.diff(warr_norm, n=1, axis=0) # first difference between normalized entries

        w_thr = np.zeros(wtable.shape[1]) # vector to hold threshold values for pixels
        w_thr[:] = np.inf # anything that's not set below will never pass threshold

        for i in range(n): # iterate over the first n positions
            w_thr[wdiff[i, :] >= diff_thr[i]] = warr_sort[i, wdiff[i, :] >= diff_thr[i]] # update threshold values if dips are identified
       
        
        out = wtable.where(wtable >= w_thr, other=0)

        if returnThresh: 
            return out, w_thr
        else: 
            return out
    

    @staticmethod
    def segmentWeights(w_img, thresh_abs=0.2, peak_footprint=2, peak_mindistnce=1):
        """ Given a weight image, perform segmentation and return the properties of the segmented regions.
            First peak_local_max is applied, followed by watershed.
            w_img: 2d or 3d numpy array
            thresh_abs: Absolute intensity threshold, a parameter to peak_local_max
            peak_footprint: the neighborhood shape to find local max. If an int, specifies the radius
                for a disk or a ball for 2d or 3d images, respectively. Otherwise, it should be 
                a numpy array specifying the neighborhood
            peak_mindistance: minimum distance between two peaks, refer to peak_local_max
        """
        wg = deepcopy(w_img.squeeze())
        ndims = len(wg.shape) # 2 or 3d    
        if type(peak_footprint) == int:
            if ndims == 2:
                ftp = disk(peak_footprint)
            elif ndims == 3:
                ftp = ball(peak_footprint)
        else:
            ftp = peak_footprint

        peaks_idx = peak_local_max(wg, min_distance=peak_mindistnce, footprint=ftp,
                               threshold_abs=thresh_abs, exclude_border=False)
        peaks = np.zeros_like(wg, dtype=bool)
        peaks[tuple(peaks_idx.T)] = True

        basins = watershed(-wg, markers=label(peaks), mask=wg>0)
        props = regionprops(basins, wg)
        properties = ('label', 'area', 'weighted_centroid', 'max_intensity', 'mean_intensity', 'equivalent_diameter', 'eccentricity')
        df = pd.DataFrame([[prop[p] for p in properties] for prop in props], columns=properties)
        if df.shape[0] == 0:
            return df
        
        if ndims == 2:
            y, x = list(zip(*df['weighted_centroid'])) # centroid weighted by the intensity image
            df.insert(0, 'z', 0)
        else:
            z, y, x = list(zip(*df['weighted_centroid'])) # centroid weighted by the intensity image
            df.insert(0, 'z', np.around(z, 1))
        df.insert(1, 'y', np.around(y, 1))
        df.insert(2, 'x', np.around(x, 1))
        df = df.rename(columns={'equivalent_diameter':'diameter',
                                'max_intensity':'weight_max',
                                'mean_intensity':'weight_mean'})
        df = df.drop(labels='weighted_centroid', axis=1)
        return df

class Codebook(xr.DataArray):
    __slots__ = ()
    @classmethod
    def readFromFile(cls, file):
        """ Each line: GENE_BARCODE"""
        barcodes_df = pd.read_csv(file, sep="_", header=None, names=['gene', 'barcode'], dtype=str)
        codebook = pd.DataFrame(barcodes_df['barcode'].str.split('').str[1:-1].tolist())
        codebook['target'] = barcodes_df['gene'] + '_' + barcodes_df['barcode']
        codebook = codebook.set_index('target')
        expanded = np.array([np.array([1 * (bc.values=='1'), 
                                        1 * (bc.values=='2'), 
                                        1 * (bc.values=='3')]) for i, bc in codebook.iterrows()])
        codebook = cls(data=expanded, dims=['target', CHN, RND], 
                       coords={'target': codebook.index.to_list(),
                               'gene': ('target', barcodes_df['gene'].to_list()),
                               'barcode': ('target', barcodes_df['barcode'])
                       })
        codebook = codebook.transpose("target", RND, CHN)
        return codebook


def deconv(int_xarr, codebook, alpha, ENargs, size=None, min_norm=0.3, elbow_thrs=[0.1, 0.1]):
    """ Convenience function for decoding"""
    dcObj_ = SparseDecoder(int_xarr, codebook, alpha, ENargs=ENargs, size=size, min_norm=min_norm)
    dcObj_.applyLasso()
    dcObj_.applyOLS(elbow_thrs=elbow_thrs)
    return dcObj_

def normAndDeconv(int_xarr, codebook, alpha, ENargs, min_norm=0.3, elbow_thrs=[0.2, 0.1], chanCoefs=None, size=None):
    """ Convenience function for decoding that normalizes intensities by chanCoefs
        int_xarr: Intensity data array, either with dims (RNDCH, spatial) or (RND, CHN, y, x)
        chanCoefs: a numpy vector. If None, then will be set to a vector of ones
    """
    # flattening the images. It's not necessary but makes the code slightly more readable
    if int_xarr.dims == (RND, CHN, 'y', 'x'):
        size = (int_xarr['x'].shape[0], int_xarr['y'].shape[0])
        int_xarr = int_xarr.stack(spatial=['y', 'x']).stack(RNDCH=[RND, CHN]).transpose('spatial', 'RNDCH')

    if int_xarr.dims == (RND, CHN, 'z', 'y', 'x'):
        size = (int_xarr['x'].shape[0], int_xarr['y'].shape[0], int_xarr['z'].shape[0])
        int_xarr = int_xarr.stack(spatial=['z', 'y', 'x']).stack(RNDCH=[RND, CHN]).transpose('spatial', 'RNDCH')
        
    # make sure chanCoefs is a row vector
    if chanCoefs is None:
        chanCoefs = np.ones((1, int_xarr.shape[1]))
    else:
        chanCoefs = np.array(chanCoefs).reshape((1, -1))

    intensities = int_xarr / chanCoefs # normalize
    dcObj = deconv(intensities, codebook, alpha, ENargs, min_norm=min_norm, size=size, elbow_thrs=elbow_thrs)
    return dcObj    

        
from scipy.spatial import cKDTree
def compareSpots(train_spots, dc_spots, maxdist=1):
    """Count the number of correctly decoded spots
            The challenge is that the (x,y) coordinate of the decoded spots may
            be slightly off due to estimating the mean from the decoded pixels 
            but it can't be too far. So to match the true spots with the decoded
            spots I will use a kdtree 1 nearest neighbor model for each gene.
    """
    dc_spots['matched'] = False # will change True if a match is found
    train_spots['matched'] = False
    train_spots['match_label'] = np.nan
    targets = train_spots['target'].unique()

    for tar in targets:
        tspots = train_spots.query("target == @tar")
        dspots = dc_spots.query('target == @tar')
        kdt = cKDTree(tspots[['x', 'y']].values)
        dists, inds = kdt.query(dspots[['x', 'y']].values, k=1)
        train_spots.loc[tspots.index[inds[dists < maxdist]], 'matched'] = True
        train_spots.loc[tspots.index[inds[dists < maxdist]], 'match_label'] = dspots.loc[dists < maxdist]['label'].to_list()
        dc_spots.loc[dspots.index[dists < maxdist], 'matched'] = True
    return train_spots, dc_spots