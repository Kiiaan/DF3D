import argparse
import io
import json
import os
import zipfile
import re
from typing import Mapping, Tuple, Union

import numpy as np
import requests
from skimage.io import imread
from slicedimage import ImageFormat
import pandas as pd

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher
from starfish.experiment.builder import write_experiment_json
from starfish.types import Axes, Coordinates, Features, Number
from shutil import copy2
from utils import getMetaData, getNumSections
import yaml

class DARTFISHTile(FetchedTile):
	def __init__(self, file_path):
		self.file_path = file_path

	@property
	def shape(self) -> Tuple[int, ...]:
		return SHAPE

	@property
	def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
		# pathRE = re.compile(r"(.*)/(.*)/MIP_\d+_.*(_FOV\d+)_.*.tif") # group(0) = whole string, group(1) = MIP_SITKaligned path, group(2) = FOV, group(3) = filename
		# pathSplit = pathRE.search(self.file_path)
		fov_dir = os.path.dirname(self.file_path)
		registered_dir = os.path.dirname(fov_dir)
		fov = os.path.basename(fov_dir)
		
		#read coordinates file
		coordinatesTablePath = stitch_coords
		
		if os.path.exists(coordinatesTablePath):
			coordinatesTable = pd.read_csv(coordinatesTablePath)
			if coordinatesTable.x.min()<0:
				coordinatesTable.x = coordinatesTable.x.subtract(coordinatesTable.x.min())
			if coordinatesTable.y.min()<0:
				coordinatesTable.y = coordinatesTable.y.subtract(coordinatesTable.y.min())
		
			#find coordinates
			locs= coordinatesTable.loc[coordinatesTable.fov == fov].reset_index(drop=True)
			#print(pathSplit.group(3))
			#print("MIP_6_dc3" + pathSplit.group(3) + "_ch03.tif")
			self.locs = {
			Coordinates.X: (locs.x[0]*VOXEL["X"], (locs.x[0] + SHAPE[Axes.X])*VOXEL["X"]),
			Coordinates.Y: (locs.y[0]*VOXEL["Y"], (locs.y[0] + SHAPE[Axes.Y])*VOXEL["Y"]),
			Coordinates.Z: (0.0, 10.0),
		}
		else:
			print("Coordinate file did not exist at: {}".format(coordinatesTablePath))
			self.locs = {
			Coordinates.X: (0.0, 0.001),
			Coordinates.Y: (0.0, 0.001),
			Coordinates.Z: (0.0, 0.001),
		}
		return self.locs

	def tile_data(self) -> np.ndarray:
		return imread(self.file_path)


class DARTFISHPrimaryTileFetcher(TileFetcher):
	def __init__(self, input_dir):
		self.input_dir = input_dir

	@property
	def ch_dict(self):
		ch_dict = {0: 'ch02', 1: 'ch00', 2: 'ch03'}
		return ch_dict

	@property
	def round_dict(self):
		round_str = RND_LIST
		round_dict = dict(enumerate(round_str))
		return round_dict

	def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
		filename = "REG_{}_FOV{}_z{:02}_{}.tif".format(self.round_dict[r],
												format_fov(fov),
												z,
												self.ch_dict[ch]
												)
		file_path = os.path.join(self.input_dir,"FOV{}".format(format_fov(fov)), filename)
		return DARTFISHTile(file_path)
#	def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
#		return DARTFISHTile(os.path.join(self.input_dir, "Subtracted","Pos{:02d}".format(fov+1),
#							"MAX_Cycle{}_Position{:02d}_ch{:02d}.tif".format(r+1, fov+1, ch)))


class DARTFISHnucleiTileFetcher(TileFetcher):
	def __init__(self, path):
		self.path = path

	def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
		file_path = os.path.join(self.path,"FOV{}".format(format_fov(fov)),"REG_{}_FOV{}_z{:02}_ch00.tif".format(RND_DRAQ5,format_fov(fov), z))
		return DARTFISHTile(file_path)

class DARTFISHbrightfieldTileFetcher(TileFetcher):
	def __init__(self, path):
		self.path = path

	def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
		file_path = os.path.join(self.path,"FOV{}".format(format_fov(fov)),"REG_{}_FOV{}_z{:02}_ch01.tif".format(RND_ALIGNED,format_fov(fov), z))
		return DARTFISHTile(file_path)


def download(input_dir, url):
	print("Downloading data ...")
	r = requests.get(url)
	z = zipfile.ZipFile(io.BytesIO(r.content))
	z.extractall(input_dir)


def write_json(res, output_path):
	json_doc = json.dumps(res, indent=4)
	print(json_doc)
	print("Writing to: {}".format(output_path))
	with open(output_path, "w") as outfile:
		json.dump(res, outfile, indent=4)


def format_data(input_dir, output_dir, fov_count, codebook_path, rounds = 6, channels = 3, zplanes = 54):
	if not input_dir.endswith("/"):
		input_dir += "/"

	if not output_dir.endswith("/"):
		output_dir += "/"

 #   if d:
 #	   url = "http://d1zymp9ayga15t.cloudfront.net/content/Examplezips/ExampleInSituSequencing.zip"
 #	   download(input_dir, url)
 #	   input_dir += "ExampleInSituSequencing/"
 #	   print("Data downloaded to: {}".format(input_dir))
 #   else:
	 #   input_dir += "ExampleInSituSequencing/"
  #	  print("Using data in : {}".format(input_dir))

	# def add_codebook(experiment_json_doc):
		# experiment_json_doc['codebook'] = "/media/Home_Raid1/rque/KPMP/scripts/codebooks/codebook_B48G_full.json"

		# return experiment_json_doc

	def overwrite_codebook(codebook_path,output_dir):
		copy2(codebook_path,os.path.join(output_dir,"codebook.json"))	
		
	# the magic numbers here are just for the ISS example data set.
	write_experiment_json(
		output_dir,
		fov_count,
		ImageFormat.TIFF,
		primary_image_dimensions={
			Axes.ROUND: rounds,
			Axes.CH: channels,
			Axes.ZPLANE: zplanes,
		},
		aux_name_to_dimensions={
			'nuclei': {
				Axes.ROUND: 1,
				Axes.CH: 1,
				Axes.ZPLANE: zplanes,
			},
			'dic': {
				Axes.ROUND: 1,
				Axes.CH: 1,
				Axes.ZPLANE: zplanes,
			},
		},
		primary_tile_fetcher=DARTFISHPrimaryTileFetcher(input_dir),
		aux_tile_fetcher={
			"nuclei": DARTFISHnucleiTileFetcher(os.path.join(input_dir)),
			"dic": DARTFISHbrightfieldTileFetcher(os.path.join(input_dir))
		},
		# postprocess_func=add_codebook,
		default_shape=SHAPE
	)
	overwrite_codebook(codebook_path,output_dir)

def format_fov(fovnum):
	n_dig = len(str(number_of_fovs))
	return str(fovnum).zfill(n_dig)
	# return str(fovnum).zfill(3)

parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

RND_LIST = params['dc_rounds']
RND_ALIGNED = params['ref_reg_cycle']
RND_DRAQ5 = params['stain_round']

if params['metadata_file'] is None:
	metadataFile = os.path.join(params['dir_data_raw'], RND_ALIGNED, 'MetaData', "{}.xml".format(RND_ALIGNED))
else:
	metadataFile = params['metadata_file']
	
npix, vox, number_of_fovs = getMetaData(metadataFile)
SHAPE = {Axes.ZPLANE: npix['3'], Axes.Y: npix['2'], Axes.X: npix['1']}
VOXEL = {"Y":vox['2'], "X":vox['1'], "Z":vox['3']}

n_zplanes = params['n_zplanes'] if params['n_zplanes'] is None else getNumSections(metadataFile)
	
input_dir = params['background_subt_dir'] if params['background_subtraction'] else params['reg_dir']
output_dir = params['starfish_dir']
codebook_path = params['codebook_path']

stitch_coords = os.path.join(params['stitch_dir'], 'registration_reference_coordinates.csv')

if not os.path.exists(codebook_path):
	raise FileNotFoundError("Codebook Not Found.")
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
format_data(input_dir, output_dir, number_of_fovs, codebook_path, rounds = params['n_rounds'], 
			channels = params['n_fluor_ch'], zplanes = n_zplanes)	

