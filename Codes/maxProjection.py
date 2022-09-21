import os, re, shutil, warnings, sys
from datetime import datetime
from ThreeDimensionalAligner import *
from utils import getMetaData
import argparse, yaml
import numpy as np 
import functools
from multiprocessing import Pool
from time import time 
import tifffile
from matplotlib import pyplot as plt 
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter


def getFiles(folder, file_regex, rnd, ch):
    allfiles = os.listdir(folder)
    selfiles = []
    for file in allfiles:
        mtch = file_regex.match(file)
        if mtch is not None:
            if (mtch.group('rndName') == rnd) and (mtch.group('ch') == ch):
                selfiles.append(os.path.join(folder, file))
    return(sorted(selfiles))


def mipFOV(fov, out_mother_dir, round_list, reg_dir, file_regex, channel_dict, method='gaussian', sigmaOrWidth=0.5):
    print(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: {} started to MIP".format(fov)))

    init_dir = os.getcwd()
    out_dir = os.path.join(os.path.abspath(out_mother_dir), fov)
    meta_dir = os.path.join(out_dir, "MetaData")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for rnd in round_list:
        in_dir = os.path.join(reg_dir, rnd)
        for ch in channel_dict[rnd]:
            imgfiles = getFiles(os.path.join(os.path.abspath(reg_dir), fov), file_regex, rnd, ch)

            # put images of correct z_range in list of array
            image_list = []
            if method == 'gaussian':
                for i, file in enumerate(imgfiles):
                    image_list.append(gaussian_filter(plt.imread(file), sigma=sigmaOrWidth))
            elif method == 'median':
                for i, file in enumerate(imgfiles):
                    image_list.append(median_filter(plt.imread(file), size=sigmaOrWidth))

            image_stack = np.dstack(image_list)
            max_array = np.amax(image_stack, axis=2)
            outname = re.sub(r"_z\d{2}", "", os.path.basename(imgfiles[0]))
            tifffile.imwrite(os.path.join(out_dir, outname), max_array, imagej=True, photometric = 'minisblack', compression="zlib")



parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

#Input Data Folder
dir_input = params['reg_dir']

#Where MIPs are written to
dir_output = params['proj_dir']
if not os.path.isdir(dir_output):
    os.makedirs(dir_output)

#rounds
rnd_list = params['reg_rounds']

# smoothing method
smooth_method = params['smooth_method']

#sigma for gaussian blur OR size (width) for median
sOrW = params['smooth_degree']

twoChRnds = params['twoChannelRounds'] 

#Number of FOVs
if params['metadata_file'] is None:
    metadataFile = os.path.join(params['dir_data_raw'], params['ref_reg_cycle'], 'MetaData', "{}.xml".format(params['ref_reg_cycle']))
else:
    metadataFile = params['metadata_file']

_, _, n_fovs = getMetaData(metadataFile)

n_pool = params['mip_npool']

channelDict = dict((rnd,['ch00', 'ch01', 'ch02', 'ch03']) if rnd not in twoChRnds else (rnd,['ch00', 'ch01']) for rnd in rnd_list)

pat3d = r"(?P<rndName>\S+)?_(?P<fov>FOV\d+)_(?P<z>z\d+)_(?P<ch>ch\d+)\S*.tif"
regex_3d = re.compile(pat3d)

partial_mip = functools.partial(mipFOV, out_mother_dir=dir_output, round_list=rnd_list, 
                                    reg_dir=dir_input, file_regex=regex_3d,
                                    channel_dict=channelDict, method=smooth_method, sigmaOrWidth=sOrW)
t1 = time()

fov_names = ["FOV" + str(n).zfill(len(str(n_fovs))) for n in range(n_fovs)]

with Pool(n_pool) as P:
    list(P.map(partial_mip, fov_names))

t2 = time()
print('Elapsed time ', t2 - t1)

