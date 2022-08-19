from typing import Optional, Tuple, cast

from starfish.core.codebook.codebook import Codebook
from starfish.core.spots.DetectPixels.combine_adjacent_features import CombineAdjacentFeatures, ConnectedComponentDecodingResult, TargetsMap
from starfish.core.spots import DetectPixels
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, Number, SpotAttributes
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from skimage.measure import label, regionprops
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import xarray as xr


NORM = 'norm'
class PixelSpotDecoder_modified(DetectPixels.PixelSpotDecoder):
    def run(
            self,
            primary_image: ImageStack,
            n_processes: Optional[int] = None,
            connectivity : Optional[int] = None,
            *args,
    ) -> Tuple[DecodedIntensityTable, ConnectedComponentDecodingResult]:
        """decode pixels and combine them into spots using connected component labeling
        Parameters
        ----------
        primary_image : ImageStack
            ImageStack containing spots
        n_processes : Optional[int]
            The number of processes to use for CombineAdjacentFeatures.
             If None, uses the output of os.cpu_count() (default = None).
        Returns
        -------
        DecodedIntensityTable :
            IntensityTable containing decoded spots
        ConnectedComponentDecodingResult :
            Results of connected component labeling
        """
        pixel_intensities = IntensityTable.from_image_stack(primary_image)
        decoded_intensities = decode_metric_modified(
            self.codebook,
            pixel_intensities,
            max_distance=self.distance_threshold,
            min_intensity=self.magnitude_threshold,
            norm_order=self.norm_order,
            metric=self.metric
        )
        caf = CombineAdjacentFeatures_modified(
            min_area=self.min_area,
            max_area=self.max_area,
            mask_filtered_features=True
        )
        decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities,
                                                        n_processes=n_processes,
                                                        connectivity=connectivity)

        transfer_physical_coords_to_intensity_table(image_stack=primary_image,
                                                    intensity_table=decoded_spots)
        return decoded_spots, image_decoding_results

        # return decoded_intensities
    
def decode_metric_modified(
        codebook, intensities: IntensityTable, max_distance: Number, min_intensity: Number,
        norm_order: int, metric: str = 'euclidean', return_original_intensities: bool = False
) -> DecodedIntensityTable:
    """
    Assigns intensity patterns that have been extracted from an :py:class:`ImageStack` and
    stored in an :py:class:`IntensityTable` by a :py:class:`SpotFinder` to the gene targets that
    they encode.
    This method carries out the assignment by first normalizing both the codes and the
    recovered intensities to be unit magnitude using an L2 norm, and then finds the closest
    code for each feature according to a distance metric (default=euclidean).
    Features greater than :code:`max_distance` from their nearest code, or that have an average
    intensity below :code:`min_intensity` are not assigned to any feature.
    Parameters
    ----------
    intensities : IntensityTable
        features to be decoded
    max_distance : Number
        maximum distance between a feature and its closest code for which the coded target will
        be assigned.
    min_intensity : Number
        minimum intensity for a feature to receive a target annotation
    norm_order : int
        the scipy.linalg norm to apply to normalize codes and intensities
    metric : str
        the sklearn metric string to pass to NearestNeighbors
    return_original_intensities: bool
        If True returns original intensity values in the DecodedIntensityTable instead of
        normalized ones (default=False)
    Notes
    -----
    The available norms for this function can be found at the following link:
    :doc:`numpy:reference/generated/numpy.linalg.norm`
    The available metrics for this function can be found at the following link:
    :doc:`scipy:reference/spatial.distance`
    Returns
    -------
    IntensityTable :
        Intensity table containing normalized intensities, target assignments, distances to
        the nearest code, and the filtering status of each feature.
    """

    codebook._validate_decode_intensity_input_matches_codebook_shape(intensities)

    # add empty metadata fields and return
    if intensities.sizes[Features.AXIS] == 0:
        return DecodedIntensityTable.from_intensity_table(
            intensities,
            targets=(Features.AXIS, np.empty(0, dtype='U')),
            distances=(Features.AXIS, np.empty(0, dtype=np.float64)),
            passes_threshold=(Features.AXIS, np.empty(0, dtype=bool)))

    # normalize both the intensities and the codebook
    norm_intensities, norms = codebook._normalize_features(intensities, norm_order=norm_order)
    norm_codes, _ = codebook._normalize_features(codebook, norm_order=norm_order)

    metric_outputs, targets = approximate_nearest_code(
        norm_codes, norm_intensities, metric=metric)

    # only targets with low distances and high intensities should be retained
    passes_filters = np.logical_and(
        norms >= min_intensity,
        metric_outputs[:, 0] <= max_distance,
        dtype=bool
    )

    return_intensities = intensities if return_original_intensities else norm_intensities
    # norm_intensities is a DataArray, make it back into an IntensityTable
    return DecodedIntensityTable_modified.from_intensity_table(
        return_intensities,
        targets=(Features.AXIS, targets[:, 0]),
        distances=(Features.AXIS, metric_outputs[:, 0]),
        second_targets=(Features.AXIS, targets[:, 1]),
        second_distances=(Features.AXIS, metric_outputs[:, 1]),
        norms=(Features.AXIS, norms), # 220810: added by Kian
        passes_threshold=(Features.AXIS, passes_filters))
    # return return_intensities, (Features.AXIS, targets), (Features.AXIS, metric_outputs), (NORM, norms), (Features.AXIS, passes_filters)

class CombineAdjacentFeatures_modified(CombineAdjacentFeatures):
    def run(
            self, intensities: IntensityTable,
            n_processes: Optional[int] = None,
            connectivity : Optional[int] = None,
    ) -> Tuple[DecodedIntensityTable, ConnectedComponentDecodingResult]:
        """
        Execute the combine_adjacent_features method on an IntensityTable containing pixel
        intensities
        Parameters
        ----------
        intensities : IntensityTable
            Pixel intensities of an imaging experiment
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter
        Returns
        -------
        IntensityTable :
            Table whose features comprise sets of adjacent pixels that decoded to the same target
        ConnectedComponentDecodingResult :
            NamedTuple containing :
                region_properties :
                    the properties of each connected component, in the same order as the
                    IntensityTable
                label_image : np.ndarray
                    An image where all pixels of a connected component share the same integer ID
                decoded_image : np.ndarray
                    Image whose pixels correspond to the targets that the given position in the
                    ImageStack decodes to.
        """

        # map target molecules to integers so they can be reshaped into an image that can
        # be subjected to a connected-component algorithm to find adjacent pixels with the
        # same targets
        targets = intensities[Features.TARGET].values
        target_map = TargetsMap(targets)

        # create the decoded_image
        decoded_image = self._intensities_to_decoded_image(
            intensities,
            target_map,
            self._mask_filtered,
        )

        # label the decoded image to extract connected component features
        conn = connectivity if not connectivity is None else self._connectivity # added by Kian
        label_image: np.ndarray = label(decoded_image, connectivity=conn)

        # calculate properties of each feature
        props: List = regionprops(np.squeeze(label_image))

        # calculate mean intensities across the pixels of each feature
        mean_pixel_traces = self._calculate_mean_pixel_traces(
            label_image,
            intensities,
        )

        # Create SpotAttributes and determine feature filtering outcomes
        spot_attributes, passes_filter = self._create_spot_attributes(
            props,
            decoded_image,
            target_map,
            n_processes=n_processes
        )

        # augment the SpotAttributes with filtering results and distances from nearest codes
        spot_attributes.data[Features.DISTANCE] = mean_pixel_traces[Features.DISTANCE]
        spot_attributes.data[Features.PASSES_THRESHOLDS] = passes_filter

        # 220810: added by Kian. Augment the SpotAttributes with barcode magnitude info
        spot_attributes.data[NORM] = mean_pixel_traces[NORM]
        
        # 220812: added by Kian. Including the second top barcode in the attributes
        spot_attributes.data['second_distance'] = mean_pixel_traces['second_distance']
        spot_attributes.data['second_target'] = mean_pixel_traces['second_target']

        # create new indexes for the output IntensityTable
        channel_index = mean_pixel_traces.indexes[Axes.CH]
        round_index = mean_pixel_traces.indexes[Axes.ROUND]
        coords = IntensityTable._build_xarray_coords(
            spot_attributes=spot_attributes,
            round_values=round_index,
            channel_values=channel_index)

        # create the output IntensityTable
        dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value)
        intensity_table = DecodedIntensityTable(
            data=mean_pixel_traces, coords=coords, dims=dims
        )

        # combine the various non-IntensityTable results into a NamedTuple before returning
        ccdr = ConnectedComponentDecodingResult(props, label_image, decoded_image)

        return intensity_table, ccdr

    @staticmethod
    def _calculate_mean_pixel_traces(
            label_image: np.ndarray,
            intensities: IntensityTable,
    ) -> IntensityTable:
        """
        For all pixels that contribute to a connected component, calculate the mean value for
        each (round, ch), producing an average "trace" of a feature across the imaging experiment
        Parameters
        ----------
        label_image : np.ndarray
            An image where all pixels of a connected component share the same integer ID
        intensities : IntensityTable
            decoded intensities
        Returns
        -------
        IntensityTable :
            an IntensityTable where the number of features equals the number of connected components
            and the intensities of each each feature is its mean trace.
        """

        pixel_labels = label_image.reshape(-1)

        # Use a pandas groupby approach-based approach, because it is much faster than xarray

        # If needed, it is possible to be even faster than pandas:
        # https://stackoverflow.com/questions/51975512/\
        # faster-alternative-to-perform-pandas-groupby-operation

        # stack intensities
        stacked = intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))

        # drop into pandas to use their faster groupby
        traces: pd.DataFrame = pd.DataFrame(
            stacked.values,
            index=pixel_labels,
            columns=stacked.traces.to_index()
        )

        #
        distances: pd.Series = pd.Series(
            stacked[Features.DISTANCE].values, index=pixel_labels
        )

        distances_2: pd.Series = pd.Series(
            stacked['second_distance'].values, index=pixel_labels
        )

        targets_2: pd.Series = pd.Series(
            stacked['second_target'].values, index=pixel_labels
        )

        norms: pd.Series = pd.Series(
            stacked[NORM].values, index=pixel_labels
        )

        grouped = traces.groupby(level=0)
        pd_mean_pixel_traces = grouped.mean()

        grouped = distances.groupby(level=0)
        pd_mean_distances = grouped.mean()

        pd_mean_norms = norms.groupby(level=0).mean()
        pd_mean_2distances = distances_2.groupby(level=0).mean()
        pd_2targets = targets_2.groupby(level=0).agg(lambda x: x.value_counts().index[0])
        
        pd_xarray = IntensityTable(
            pd_mean_pixel_traces,
            dims=(Features.AXIS, 'traces'),
            coords=dict(
                traces=('traces', pd_mean_pixel_traces.columns),
                distance=(Features.AXIS, pd_mean_distances),
                norm=(Features.AXIS, pd_mean_norms),
                second_target=(Features.AXIS, pd_2targets), 
                second_distance=(Features.AXIS, pd_mean_2distances),
                features=(Features.AXIS, pd_mean_pixel_traces.index)
            )
        )
        mean_pixel_traces = pd_xarray.unstack('traces')

        # the 0th pixel trace corresponds to background. If present, drop it.
        try:
            mean_pixel_traces = mean_pixel_traces.drop_sel({Features.AXIS: 0})
        except KeyError:
            pass

        return cast(IntensityTable, mean_pixel_traces)

class DecodedIntensityTable_modified(DecodedIntensityTable):
    @classmethod
    def from_intensity_table(
            cls,
            intensities: IntensityTable,
            targets: Tuple[str, np.ndarray],
            distances: Optional[Tuple[str, np.ndarray]] = None,
            second_targets : Optional[Tuple[str, np.ndarray]] = None,
            second_distances: Optional[Tuple[str, np.ndarray]] = None, 
            norms: Optional[Tuple[str, np.ndarray]] = None,
            passes_threshold: Optional[Tuple[str, np.ndarray]] = None,
            rounds_used: Optional[Tuple[str, np.ndarray]] = None):

        """
        Assign target values to intensities.
        Parameters
        ----------
        intensities : IntensityTable
            intensity_table to assign target values to
        targets : Tuple[str, np.ndarray]
            Target values to assign
        distances : Optional[Tuple[str, np.ndarray]]
            Corresponding array of distances from target for each feature
        passes_threshold : Optional[Tuple[str, np.ndarray]]
            Corresponding array of boolean values indicating if each itensity passed
            given thresholds.
        rounds_used: Optional[Tuple[str, np.ndarray]]
            Corresponding array of integers indicated the number of rounds this
            decoded intensity was found in
        Returns
        -------
        DecodedIntensityTable
        """

        intensities = cls(intensities)
        intensities[Features.TARGET] = targets
        if distances:
            intensities[Features.DISTANCE] = distances
        if second_targets:
            intensities['second_target'] = second_targets
        if second_distances:
            intensities['second_distance'] = second_distances
        if norms:
            intensities[NORM] = norms    
        if passes_threshold:
            intensities[Features.PASSES_THRESHOLDS] = passes_threshold
        if rounds_used:
            intensities['rounds_used'] = rounds_used
        return intensities


def approximate_nearest_code(
            norm_codes: "Codebook", norm_intensities: xr.DataArray, metric: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """find the nearest code for each feature using the ball_tree approximate NN algorithm
        Parameters
        ----------
        norm_codes : Codebook
            codebook with each code normalized to unit length (sum = 1)
        norm_intensities : IntensityTable
            intensity table with each feature normalized to unit length (sum = 1)
        metric : str
            the sklearn metric string to pass to NearestNeighbors
        Returns
        -------
        np.ndarray : metric_output
            the output of metric applied to each feature closest code
        np.ndarray : targets
            the gene that corresponds to each matched code
        Notes
        -----
        This function does not verify that the intensities have been normalized.
        """
        linear_codes = norm_codes.stack(traces=(Axes.CH.value, Axes.ROUND.value)).values
        linear_features = norm_intensities.stack(
            traces=(Axes.CH.value, Axes.ROUND.value)).values

        # reshape into traces
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric=metric).fit(linear_codes)
        metric_output, indices = nn.kneighbors(linear_features)
        gene_ids = (norm_codes.indexes[Features.TARGET].values[indices])

        return metric_output, gene_ids