import os, re, shutil, warnings, sys
from datetime import datetime
from ThreeDimensionalAligner import *
from utils import getMetaData
import argparse, yaml
import numpy as np 
import functools
from multiprocessing import Pool
from time import time 
import logging
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    """From here:
    https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings"""
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def getFiles(folder, file_regex, fov, ch):
    allfiles = os.listdir(folder)
    selfiles = []
    for file in allfiles:
        mtch = file_regex.search(file)
        if mtch is not None:
            if (mtch.group('fov') == fov) and (mtch.group('ch') == ch):
                selfiles.append(os.path.join(folder, file))
    return(sorted(selfiles))


def alignFOV(fov, out_mother_dir, round_list, raw_dir, channel_DIC, cycle_other, channel_DIC_other, ref_rnd, maxIter, numResolution, file_regex, channel_dict, voxelInfo, AutomaticScale, Scales):
    # print(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: FOV{} started to align".format(fov[1:])))
    logging.info(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: FOV{} started to align".format(fov[1:])))

    init_dir = os.getcwd()
    fov_fov = "FOV{}".format(fov[1:])
    out_dir = os.path.join(os.path.abspath(out_mother_dir), fov_fov)
    meta_dir = os.path.join(out_dir, "MetaData")

    Path(out_dir).mkdir(exist_ok=True)
    Path(meta_dir).mkdir(exist_ok=True)

    os.chdir(out_dir)

    for mov_rnd in round_list:
        logging.info("FOV{}_round {}".format(fov[1:], mov_rnd))
        in_dir = os.path.join(raw_dir, ref_rnd)
        
        # find the transform parameters for this (mov_rnd, FOV) pair
        chan_ref = channel_DIC if ref_rnd not in cycle_other else channel_DIC_other[ref_rnd]
        ref_files = getFiles(os.path.join(os.path.abspath(raw_dir), ref_rnd), file_regex, fov, chan_ref)

        chan_mov = channel_DIC if mov_rnd not in cycle_other else channel_DIC_other[mov_rnd]
        mov_files = getFiles(os.path.join(os.path.abspath(raw_dir), mov_rnd), file_regex, fov, chan_mov)

        ### move the reference images directory to the output
        # if mov_rnd == ref_rnd:
        #     for ch in channel_dict[mov_rnd]:
        #         in_files = getFiles(os.path.join(os.path.abspath(raw_dir), mov_rnd), file_regex, fov, ch)
        #         in_bnames = [os.path.basename(f) for f in in_files]
        #         out_bnames = [re.sub(r"_s(\d+)_", r"_FOV\1_", ib) for ib in in_bnames]
        #         out_bnames = ["REG_" + ob for ob in out_bnames]
        #         out_files = [os.path.join(out_dir, ob) for ob in out_bnames]
        #         for in1, out1 in zip(in_files, out_files):
        #             shutil.copy2(src=in1, dst=out1)
        #     continue

        aligner = ThreeDimensionalAligner(mov_files, ref_files, transform = "rigid", NumberOfResolutions = numResolution, 
                                            MaximumNumberOfIterations = maxIter, NumberOfSpatialSamples = int(4000 * len(mov_files) **0.5), 
                                            transParamFile = os.path.join(meta_dir, "{}-to-{}_transformParameters.txt".format(mov_rnd, ref_rnd)),
                                            voxelSize = voxelInfo, AutomaticScale = AutomaticScale, Scales = Scales)
        try:
            aligner.findTransformParameters()
            for ch in channel_dict[mov_rnd]:
                in_files = getFiles(os.path.join(os.path.abspath(raw_dir), mov_rnd), file_regex, fov, ch)
                in_bnames = [os.path.basename(f) for f in in_files]
                out_bnames = [re.sub(r"_s(\d+)_", r"_FOV\1_", ib) for ib in in_bnames]
                out_bnames = ["REG_" + ob for ob in out_bnames]
                out_files = [os.path.join(out_dir, ob) for ob in out_bnames]
                aligner.applyTransformation([in_files], [out_files])

                # move the metadata file to the output directory
                metaFile = os.listdir(os.path.join(os.path.abspath(raw_dir), mov_rnd, 'MetaData'))
                metaFile = [f for f in metaFile if not re.search("{0}.xml".format(mov_rnd), f) is  None]
                metaFile = os.path.join(os.path.abspath(raw_dir), mov_rnd, 'MetaData', metaFile[0])
                # metaFile = os.path.join(os.path.abspath(raw_dir), mov_rnd, 'MetaData', "{0}.xml".format(mov_rnd))
                # metaFile = [os.path.join(os.path.abspath(raw_dir), mov_rnd, 'MetaData', f) for f in os.listdir()]#"{0}.xml".format(mov_rnd))]
                if os.path.isfile(metaFile):
                    if not os.path.exists(os.path.join("MetaData")):
                        os.mkdir(os.path.join("MetaData"))

                    shutil.copy2(src = metaFile, 
                        dst = os.path.join('MetaData', "{0}.xml".format(mov_rnd)))
                else:
                    logging.info("MetaData file wasn't found at {}".format(os.path.join(os.path.abspath(raw_dir), mov_rnd, 'MetaData', "{0}.xml".format(mov_rnd))))
            
        except Exception as exc:
            logging.error("Exception raised in FOV{}_round {}".format(fov[1:], mov_rnd))
            err_logger.error("FOV{}\t{}".format(fov[1:], mov_rnd))
            print('Exception for FOV{}, {}:'.format(fov[1:], mov_rnd))
            print(exc)


    os.chdir(init_dir)
    logging.info(datetime.now().strftime("%Y-%d-%m_%H:%M:%S: FOV{} done".format(fov[1:])))


parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

#Raw Data Folder
dir_data_raw = params['dir_data_raw']

#Where Aligned MIPs are written to
dir_output_aligned = params['reg_dir']
if not os.path.isdir(dir_output_aligned):
    os.makedirs(dir_output_aligned)

#rounds
rnd_list = params['reg_rounds']

# setup loggings
logging.basicConfig(filename=os.path.join(dir_output_aligned, datetime.now().strftime("%Y-%d-%m_%H-%M_3dAlignment.log")),
                    level=logging.INFO)
err_logger = setup_logger(name='error_logs', 
                          log_file=os.path.join(dir_output_aligned, datetime.now().strftime("%Y-%d-%m_%H-%M_3dAlignment_errors.log")),
                          level=logging.ERROR)
# smoothing method
smooth_method = params['smooth_method']

#sigma for gaussian blur OR size (width) for median
sOrW = params['smooth_degree']

reference_cycle = params['ref_reg_cycle'] # Which cycle to align to
channel_DIC = params['ref_reg_ch'] # DIC channel for reference cycle
cycle_other = list(params['cycle_other']) # if there are other data-containing folders which need to be aligned but are not names "CycleXX"
channel_DIC_other = params['cycle_other'] # DIC channel for other data-containing folders
twoChRnds = params['twoChannelRounds'] if not params['twoChannelRounds'] is None else []

# channelDict = dict((rnd,['ch00', 'ch01', 'ch02', 'ch03']) if rnd not in twoChRnds else (rnd,['ch00', 'ch01']) for rnd in rnd_list)
channelDict = dict((rnd,['ch00', 'ch01', 'ch02', 'ch03']) if rnd not in twoChRnds else (rnd,['ch00', 'ch01']) for rnd in rnd_list)

#Number of FOVs
if params['metadata_file'] is None:
    metadataFile = os.path.join(params['dir_data_raw'], reference_cycle, 'MetaData', "{}.xml".format(reference_cycle))
else:
    metadataFile = params['metadata_file']

_, voxelSpacing, n_fovs = getMetaData(metadataFile)
voxelSpacing = tuple([0.001 * voxelSpacing[i] for i in voxelSpacing]) # convert the dict to a tuple with mm units

n_pool = params['reg_npool']

maxIter = params['reg_maxIter'] # maximum number of iteractions for registration
numResolution = params['NumberOfResolutions'] # number of resolution pyramids 
transform = params['reg_transform'] # translation, rigid, or affine
autoScale = params['AutomaticScaleEstimation'] # binary. If params scales are to be estimated by elastix
param_scales = params['params_scales'] # scale of parameters set manually

pat3d = params['filePat3d'].format("s")
regex_3d = re.compile(pat3d)


partial_align = functools.partial(alignFOV, out_mother_dir=dir_output_aligned, round_list=rnd_list, 
                                    raw_dir=dir_data_raw, channel_DIC=channel_DIC, cycle_other=cycle_other, channel_DIC_other=channel_DIC_other, 
                                    ref_rnd=reference_cycle, maxIter=maxIter, numResolution=numResolution, file_regex=regex_3d,
                                    channel_dict=channelDict, voxelInfo=voxelSpacing, AutomaticScale = autoScale, Scales=param_scales)
# partial_align = functools.partial(alignFOV, out_mother_dir=dir_output_aligned, round_list=['dc2', 'dc3'], 
#                                     raw_dir=dir_data_raw, channel_DIC=channel_DIC, cycle_other=cycle_other, channel_DIC_other=channel_DIC_other, 
#                                     ref_rnd=reference_cycle, maxIter=maxIter, numResolution=numResolution, file_regex=regex_3d,
#                                     channel_dict=channelDict, voxelInfo=voxelSpacing, AutomaticScale = autoScale, Scales=param_scales)

t1 = time()

fov_names = ["s" + str(n).zfill(len(str(n_fovs))) for n in range(n_fovs)]
with Pool(n_pool) as P:
    list(P.map(partial_align, fov_names))

t2 = time()
logging.info('Elapsed time {}'.format(t2 - t1))

