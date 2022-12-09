import numpy as np, xarray as xr, pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from copy import deepcopy
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball

RND, CHN = 'rnd', 'ch'

class Decoder2D():
    def __init__(self, intensities, codebook, alpha, size=None, verbose=False, **kwargs):
        """intensities: xr.DataArray. dims either
            1) (RND, CHN, y, x)
            2) (spatial, RNDCH). ُُThis shape can be obtained by using DataArray.stack()
        """
        self.ints = intensities
        self.alpha = alpha
        self.cb = codebook
        self.ls = self._setupLasso(**kwargs)
        self.verbose = verbose
        self.norms = None # will be set during prepping pixels
        self.size = size # size of the field. Is necessary only if result image is going to be created
        self.lasso_pixs = None # will be set during prepping pixels
#         w3d = None # will be set in createResultImage
        
    def prepTrainingPixels(self, min_norm=0.5):
        """ Finding pixels that the lasso model will be trained on. The output, self.lasso_pixs is of the form ('spatial', 'RNDCH')"""
        if self.ints.dims == (RND, CHN, 'y', 'x'):
            pixel_intensities = self.ints.stack(spatial=['y', 'x']).stack(RNDCH=[RND, CHN]).transpose('spatial', 'RNDCH')
            self.size = (self.ints['x'].shape[0], self.ints['y'].shape[0])
        elif self.ints.dims == ('spatial', 'RNDCH'):
            pixel_intensities = deepcopy(self.ints)
        else: 
            raise ValueError("Incorrect dimensions for self.ints: {}".format(self.ints.dims))
        self.norms = np.linalg.norm(pixel_intensities.values, ord=2, axis=1)
        self.lasso_pixs = pixel_intensities[self.norms >= min_norm]
        
    def applyLasso(self):
        if self.lasso_pixs is None:
            raise ValueError("training pixels aren't set yet!")
        
        cb_flat = self.cb.stack(flatcode = (RND, CHN))
        
        if self.lasso_pixs.shape[0] == 0:   # no pixels with enough fluorescence
            if self.verbose:
                print("No pixels with enough fluorescence. Skipping deconvolution")
            self.lasso_table = None
            return

        """ Fitting the Lasso model"""
        if self.verbose:
            print("Starting the lasso fit. Data shape: {}".format(self.lasso_pixs.shape))
        weights = np.array(list(map(lambda row: self.myfit(row, cb_flat, self.ls), self.lasso_pixs)))
        self.lasso_table = xr.DataArray(weights.T,
                                        coords={'x':('pixels', self.lasso_pixs.coords['x'].values),
                                              'y':('pixels', self.lasso_pixs.coords['y'].values),
    #                                           'z':('pixels', self.lasso_pixs.coords['z'].values),
                                              'target':('codes', cb_flat.coords['target'].values),
                                              'gene' : ('codes', list(map(lambda x: x.split('_')[0], cb_flat.coords['target'].values)))}, 
                                        dims=['codes', 'pixels'])
        if self.verbose:
            print("Done fitting lasso")

    def applyOLS(self, knee_thrs=(0.1, 0.1)):
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
        w_numpy = self.kneeFilter(w_table, abs_thr=knee_thrs).values # select barcodes with lasso. weight will be updated with ols weights
        for j in range(w_table.shape[1]): # iterate over every pixel 
            selinds = np.nonzero(w_numpy[:, j])[0] # non-zero weight indices (after filtering)
            if selinds.shape[0] == 0:
                continue
            selbc = cb_flat[selinds] # the barcodes with non-zero weights
            wj = self.myfit(self.ols_pixs[j], selbc, ols) # the ols fit
            w_numpy[selinds, j] = wj # replacing the lasso weights with ols weights
        self.ols_table = xr.DataArray(w_numpy, dims = w_table.dims,
                                      coords = w_table.coords)
        
    def createResultImage(self, w_table, size=None):
        if self.size is None:
            raise ValueError("Field size is not set and w3d cannot be constructed")
        else:
            n_x = self.size[0]
            n_y = self.size[1]

        # if not size is None:
        #     n_x = size[0]
        #     n_y = size[1]
        # else:
        #     n_y = self.ints['y'].shape[0]
        #     n_x = self.ints['x'].shape[0]
        if w_table is None:
            w3d = xr.DataArray(np.zeros((self.cb.shape[0], n_y, n_x)), dims=['bc', 'y', 'x'], #dims=['bc', 'z', 'y', 'x'], 
                               coords = {'target' : ('bc', self.cb.target.values),
                                        'gene' : ('bc', list(map(lambda x: x.split('_')[0], self.cb['target'].values))),
                                        'x': range(n_x), 'y': range(n_y)}) # , 'z': range(n_z)
        else:
            w3d = xr.DataArray(np.zeros((len(w_table.codes), n_y, n_x)), dims=['bc', 'y', 'x'], #dims=['bc', 'z', 'y', 'x'], 
                               coords = {'target' : ('bc', w_table.codes.target.values),
                                        'gene' : ('bc', list(map(lambda x: x.split('_')[0], w_table['target'].values))),
                                        'x': range(n_x), 'y': range(n_y)}) # , 'z': range(n_z)
            w3d[:, w_table['y'], w_table['x']] = w_table
        return w3d
    
    def myfit(self, int_row, cdbook, model):
        """ int_row: xarray row of intensities
            cdbook: xarray of the codebook with flattened (onehot) barcodes
            model: a sklearn model
        """
        return model.fit(cdbook.values.T, int_row.values.reshape(-1)).coef_

    def _setupLasso(self, **kwargs):
        if "positive" in kwargs:
            positive = kwargs['positive']
        else:
            positive = True
        if "fit_intercept" in kwargs:
            fit_intercept = kwargs['fit_intercept']
        else:
            fit_intercept = False
            
        return Lasso(alpha=self.alpha, positive=positive, fit_intercept=fit_intercept)
            
    def getResultImage(self, method='lasso'):
        if method == 'lasso':
            return self.createResultImage(self.lasso_table)
        elif method == 'ols':
            return self.createResultImage(self.ols_table)
        else:
            raise ValueError('No such method {} implemented'.format(method))  

    def createSpotTable(self, w_table, flat_filter = 'topN', volume_filter=None,
                        flat_filter_kwargs={}, volume_filter_kwargs={}, 
                        thresh_abs=0.2, peak_footprint=2, peak_mindistnce=1):
        """ Filtering the weights, applying watershed segmentation to each barcode, 
                and summarizing the segmented regions. First flat filter is applied, then volume filter
            flat_filter: A function that accepts a flat (stacked) array like self.lasso_table and 
                returns an array with the same shape and type. "topN" uses the topN filter.
            volume_filter: A function that accepts a volume like w3d, and 
                returns a volume of the same shape and type.
            n: The n value for the topN filter if volume_filter=='topN'
            thresh_abs, peak_footprint and peak_mindistnce are inputs to self.segmentWeights
                (technically inputs to peak_local_max)
        """
        if self.verbose:
            print("Performing flat filtering on weights")
        if not flat_filter is None:
            if flat_filter == 'topN':
                ffilt = self.topNFilter
            elif flat_filter == 'knee':
                ffilt = self.kneeFilter
            else: 
                ffilt = flat_filter
        else:
            ffilt = lambda x: x
        w_table_filt = ffilt(deepcopy(w_table), **flat_filter_kwargs)
            
        w3d = self.createResultImage(w_table_filt)
        
        if not volume_filter is None:
            if self.verbose:
                print("Performing volume filtering on weights")
            if volume_filter == 'topN':
                w3d = self.topNFilter(w3d, **volume_filter_kwargs)
            else:
                w3d = volume_filter(w3d, **volume_filter_kwargs)
        
        if self.verbose:
            print("Performing watershed segmentation on every barcode")
        self.dc_spots = self.segmentAllTargets(w3d, thresh_abs=thresh_abs,
                                                peak_footprint=peak_footprint, peak_mindistnce=peak_mindistnce)
        return deepcopy(self.dc_spots)
        
    @staticmethod
    def topNFilter(weights, n=2):
        """ For every spatial location, keeps the top n barcode weights, fills the rest with 0
            weights: xarray with dimensions bc, z, y, x"""
        w_ntop = np.sort(weights.values, axis=0)[-n] # second to highest weight
        return weights.where(weights >= w_ntop[None], 0) #where the filtering happens
    
    @staticmethod
    def kneeFilter(wtable, n=2, abs_thr = [0.1, 0.1], returnThresh=False):
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
            w_thr[warr_norm[i+1, :] <= abs_thr[i]] = warr_sort[i, warr_norm[i+1, :] <= abs_thr[i]] # update threshold values if dips are identified
       
        
        out = wtable.where(wtable >= w_thr, other=0)

        if returnThresh: 
            return out, w_thr
        else: 
            return out

    @staticmethod
    def kneeFilter_old(wtable, n=2, diff_thr = [0.5, 0.4], returnThresh=False):
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
    def segmentAllTargets(w3d, thresh_abs=0.2, peak_footprint=2, peak_mindistnce=1):
        spots = []
        for tar in w3d['target'].values:
            wg = w3d[w3d['target'] == tar].values
            spots_g = Decoder2D.segmentWeights(wg, thresh_abs=thresh_abs, peak_footprint=peak_footprint, peak_mindistnce=peak_mindistnce)
            if spots_g.shape[0] == 0:
                continue
            spots_g['target'] = tar
            spots_g['gene'] = spots_g['target'].str.split('_').str[0]
            spots_g['label'] = tar + "_" + spots_g['label'].astype(str)
            spots.append(spots_g)
        if len(spots) > 0:
            spots = pd.concat(spots, ignore_index=True)
        else:
            spots = pd.DataFrame()
        return spots    
    
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

        y, x = list(zip(*df['weighted_centroid'])) # centroid weighted by the intensity image
        df.insert(0, 'x', np.around(x, 1))
        df.insert(1, 'y', np.around(y, 1))
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
                       coords={'target': codebook.index.to_list()})
        codebook = codebook.transpose("target", RND, CHN)
        return codebook

        
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