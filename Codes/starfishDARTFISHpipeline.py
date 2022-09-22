import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

import starfish
from starfish import Experiment
from starfish import display
from starfish import data, FieldOfView
from starfish.types import Features, Axes

from starfish import IntensityTable

from starfish.image import Filter
# from starfish.spots import DetectPixels
from starfish.core.types import Levels

from datetime import datetime
from multiprocessing import Pool 	# Kian: added 210602
import functools	# Kian: added 210602
import yaml
import argparse
from utils import getMetaData
import starfish_amend
from scipy.ndimage import median_filter
import logging

def DARTFISH_pipeline(fov, name, codebook, magnitude_threshold, binarize, min_cutoff = 0, normalize_max = None, area_threshold = (1, 100)):
	''' if normalize_max not None, then all images are linearly normalize by normalize_max '''
	imgs = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)
	logging.info(datetime.now().strftime('%Y-%d-%m_%H:%M:%S: Started field of view {}'.format(name)))

	# gauss_filt = Filter.GaussianLowPass(0.3, True)
	gauss_imgs = imgs#gauss_filt.run(imgs)
	
	sc_filt = Filter.Clip(p_min=0, p_max=100, level_method=Levels.SCALE_BY_CHUNK, is_volume=True) # right now, it doesn't do anything
	norm_imgs = sc_filt.run(gauss_imgs)

	z_filt = Filter.ZeroByChannelMagnitude(thresh=.05, normalize=binarize)
	norm_imgs = z_filt.run(norm_imgs)

	norm_imgs = norm_imgs.apply(median_filter, size=2, group_by={Axes.ROUND, Axes.CH})

	if not normalize_max is None:
		norm_imgs = norm_imgs.apply(lambda x: x * (x > min_cutoff/255))
		norm_imgs = norm_imgs.apply(lambda x: 255 / normalize_max * np.clip(x, 0, normalize_max/255)) # 211019

	# def compute_magnitudes(stack, norm_order=2):
	# 	pixel_intensities = IntensityTable.from_image_stack(stack)
	# 	feature_traces = pixel_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
	# 	norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)
	# 	return norm

	mags = None #compute_magnitudes(filtered_imgs)
	
	# how much magnitude should a barcode have for it to be considered by decoding? this was set by looking at
	# the plot above
#	magnitude_threshold = 0.2
	# how big do we expect our spots to me, min/max size. this was set to be equivalent to the parameters
	# determined by the Zhang lab.
	# area_threshold = (3, 45)
	# how close, in euclidean space, should the pixel barcode be to the nearest barcode it was called to?
	# here, I set this to be a large number, so I can inspect the distribution of decoded distances below
	
	distance_threshold = 2
	
	psd = starfish_amend.PixelSpotDecoder_modified(
		codebook=codebook,
		metric='euclidean',
		distance_threshold=distance_threshold,
		magnitude_threshold=magnitude_threshold,
		min_area=area_threshold[0],
		max_area=area_threshold[1]
	)

	spot_intensities, results = psd.run(norm_imgs)
	spot_intensities = IntensityTable(spot_intensities.where(spot_intensities[Features.PASSES_THRESHOLDS], drop=True))
	# reshape the spot intensity table into a RxC barcode vector
	pixel_traces = spot_intensities.stack(traces=(Axes.ROUND.value, Axes.CH.value))

	# extract dataframe from spot intensity table for indexing purposes
	pixel_traces_df = pixel_traces.to_features_dataframe()
	pixel_traces_df['area'] = np.pi*pixel_traces_df.radius**2
	logging.info(datetime.now().strftime('%Y-%d-%m_%H:%M:%S: Finished field of view {}'.format(name)))
	return pixel_traces_df, mags


def process_experiment(experiment: starfish.Experiment, output_dir, magnitude_threshold, binarize, normalize_max = None, 
	area_threshold = (1, 100), min_cutoff=0):
	''' if normalize_max not None, then all images are linearly normalize by normalize_max '''
	decoded_intensities = {}
	regions = {}
	df_pipe_partial = functools.partial(DARTFISH_pipeline, codebook=experiment.codebook, magnitude_threshold = magnitude_threshold, 
			binarize=binarize, normalize_max = normalize_max, area_threshold = area_threshold, min_cutoff=min_cutoff)

	# fovs = [fov for (name_, fov) in experiment.items()]
	# names = [name_ for (name_, fov) in experiment.items()]
	names, fovs = zip(*experiment.items())
	with Pool(dc_npool) as p:
		traces_dfs, mags = zip(*p.starmap(df_pipe_partial, zip(list(fovs), list(names))))
	# traces_dfs, mags = zip(*map(df_pipe_partial, list(fovs)[:2]))
	# print(traces_dfs)
	
	for name, trace in zip(names, traces_dfs):
		newName = "FOV{}".format(format_fov(int(name[4:])))
		trace.to_csv(os.path.join(output_dir,'starfish_table_bcmag_{0}_{1}'.format(magnitude_threshold,newName) + '.csv'))
	# 	print(datetime.now().strftime('%Y-%d-%m_%H:%M:%S: Finished Processing FOV {:02d} with Barcode Magnitude threshold {}'.format(count,magnitude_threshold)))
	# 	count += 1
		#decoded_intensities[name_] = decoded
		#regions[name_] = segmentation_results
#	return decoded_intensities, regions

def format_fov(fovnum):
	n_dig = len(str(number_of_fovs))
	return str(fovnum).zfill(n_dig)

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))


dc_npool = params['dc_npool']
data_dir = params['starfish_dir']
output_dir = params['dc_out']
min_intensity = params['min_intensity']
normalize_ceiling = params['max_intensity'] # set to None in order to not normalize the maximum of each image
rolonyArea = params['rolony_area']
bcmag = params['bcmag']
ifBinarize = params['dc_binarize']

if params['metadata_file'] is None:
	metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
else:
	metadataFile = params['metadata_file']

_, _, number_of_fovs = getMetaData(metadataFile)


exp = Experiment.from_json(os.path.join(data_dir,"experiment.json"))

for magnitude_threshold in [bcmag]:
	output_path = output_dir + "_bcmag{}".format(magnitude_threshold)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	# setup loggings
	logging.basicConfig(filename=os.path.join(output_path, datetime.now().strftime("%Y-%d-%m_%H-%M_3dDecoding.log")),
                    level=logging.INFO)
	
	logging.info(datetime.now().strftime('%Y-%d-%m_%H:%M:%S: Started Processing Experiment with Barcode Magnitude threshold ' + str(magnitude_threshold)))
	process_experiment(exp, output_path, magnitude_threshold,binarize=ifBinarize, min_cutoff = min_intensity, 
						normalize_max = normalize_ceiling, area_threshold = rolonyArea)
	logging.info(datetime.now().strftime('%Y-%d-%m_%H:%M:%S: Finished Processing Experiment with Barcode Magnitude threshold ' + str(magnitude_threshold)))
	
