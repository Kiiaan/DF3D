import os, re, IJ_stitch_201020 as IJS
import shutil, sys
from datetime import datetime
import pandas as pd, numpy as np
from utils import getTileLocs, plotFOVMap, getMetaData
import yaml
import argparse 
from skimage.io import imread

""" We want to stitch all channels of all cycles of DART-FISH.
    Since all the images that need to be stitched have to in the same directory, 
    we have to move images of different FOVs in the same directory and run the image stitching.
    As of now (Oct 14th, 2020), after maximum projecting and registering, images of the same FOV 
    are kept in the same directory.
    This code assumes that all images are registered, so one specified cycle and channel is
    used to find the tile configuration and that setting will be applied to all other tiles and channels. 
    IMPORTANT: The ImageJ path has to be set. 
"""
  
def add2dict2dict(key, value, dic):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]

def copy2dir(files2copy, dest_dir):
    for infile in files2copy:
        shutil.copy2(infile, dest_dir)


def createTileConfig(tileNames, tileLocs, outfile):
    """ tileNames is a list of filenames for the tiles. tileLocs is a list of tuples of the form:
        (x, y) in which x and y are the location of the tile in pixels
    """
    with open(outfile, 'w') as writer: 
        writer.writelines("# Define the number of dimensions we are working on\ndim = 2\n\n")
        writer.writelines("# Define the image coordinates\n")
        for file, (x, y) in zip(tileNames, tileLocs):
            line = "{0}; ; ({1}, {2})\n".format(file, x, y)
            writer.writelines(line)


def changeTileConfig(reffile, nrefile, nrefNames, fov_pat):
    """ Looping through all lines of the reference tile config, and change the channel
        to match the non-reference image files.
        reffile: path to the reference tile configuration file
        nrefile: path to the tile configation file that we want to generate
        nrefNames: file names that need to be substituted for the original reference filenames.
        fov_pat: regex pattern that specifies the FOV.
    """
    with open(nrefile, 'w') as writer, open(reffile, 'r') as reader:
        for line in reader:
            refmtch = re.search(".tif", line) # assuming all images are .tif
            if refmtch is None:
                writer.writelines(line)
            else:
                fov = re.search(fov_pat, line).group(0) # the FOV in this line
                
                # finding the non-ref image with the same fov
                for nrefn in nrefNames:
                    if fov in nrefn:
                        # substituting the whole file name section
                        writer.writelines(re.sub(r"^\S+.tif", nrefn, line))
#                 writer.writelines(re.sub(ref_ch, nrf_ch, line))    

                
def cleanUpImages(file_dict, file_dir):
    """ Deleted the images we moved for stitching """
    for key in file_dict:
        for file in file_dict[key]:
            os.remove(os.path.join(file_dir, os.path.basename(file)))
            
def writeReport(spOut):
    dtn = datetime.now()
    dtn = "{0}-{1}-{2}_{3}:{4}:{5}".format(dtn.year, dtn.month, dtn.day,
                                         dtn.hour, dtn.minute, dtn.second)
    print("{0}: ImageJ's stdout:".format(dtn))
    print(spOut.stdout)
    print("{0}: ImageJ's stderr:".format(dtn))
    print(spOut.stderr)    
    
def readStitchInfo(infoFile, rgx):
    """ read ImageJ's stitching output and spit out the top left position of
    each tile image on the stitched image in a dataframe"""
    with open(infoFile, 'r+') as reader:
        infoDict = {}
        for line in reader:
            if line.startswith('# Define the image coordinates'):
                break
        
        positions, xs, ys = [], [], []
        for line in reader:
            pos_re = re.search(rgx, line)
            positions.append(pos_re.group('fov'))
            
            coord_re = re.search(r".tif.*\(([-+]?[0-9]*[.][0-9]*)" + 
                                   r".*?([-+]*[0-9]*[.][0-9]*)\)", line)
            
            xs.append(float(coord_re.group(1)))
            ys.append(float(coord_re.group(2)))

        return pd.DataFrame({'fov' : positions, 'x' : xs, 'y' : ys})


parser = argparse.ArgumentParser()
parser.add_argument('param_file')
args = parser.parse_args()
params = yaml.safe_load(open(args.param_file, "r"))

# if background subtracted data available, stitch that; otherwise stitch registered data    
if params['background_subtraction']:
    data_dir = params['background_subt_dir']
else:
    data_dir = params['proj_dir']

stitch_dir = params['stitch_dir']

""" names of rounds to be stitched"""
if not params['stch_rounds'] is None:
    rounds = params['stch_rounds']
else:
    rounds = params['reg_rounds']

stitchRef = params['ref_reg_cycle'] if params['stitchRefCycle'] is None else params['stitchRefCycle'] # the round to be used as the reference for stitching
stitchChRef = params['ref_reg_ch'] if params['stitchChRef'] is None else params['stitchChRef'] # default reference channel for stitching

""" Getting the nominal tile location of the reference round using the metadata file"""
if params['metadata_file'] is None:
    metadataFile = os.path.join(params['dir_data_raw'], stitchRef, 'MetaData', "{}.xml".format(stitchRef))
else:
    metadataFile = params['metadata_file']
    
tileLocs = getTileLocs(metadataFile)


""" Stitching Inputs"""
# IJS.IJ_Stitch.getImageJ('/media/Home_Raid1_Voyager/kian/Codes/DART-FISH/image_stitching') # can be used to download ImageJ the first time
ij_path = params['ij_path']

fovs = [file for file in os.listdir(data_dir) if re.match(params['fov_pat'], file)]
fovs = sorted(fovs, key = lambda x: int(x[3:]))

if not os.path.isdir(stitch_dir):
    os.mkdir(stitch_dir)


""" Regex that explains the file names. Group "ch" is important to be set. """
# filePattern = r"(?P<intro>\S+)?_(?P<rndName>\S+)_(?P<fov>FOV\d+)_(?P<ch>ch\d+)\S*.tif$" # 0: all, 1: MIP_rnd#, 2:dc/DRAQ, 3: FOV, 4: chfile_regex = re.compile(filePattern)
filePattern = params['stchFilePattern']
fov_pat = params['fov_pat'] # pattern to extract the fov# 
fov_sub = r"\1{" + ''.join(len(str(len(fovs))) * ['i']) + "}" # string to substitute the fov# with {ii} or {iii}


""" Redirecting stdout to write in a report file"""
dtn = datetime.now()
reportfile = os.path.join(stitch_dir, "{0}-{1}-{2}_{3}:{4}:{5}-stitch report.txt".format(dtn.year, 
                                    dtn.month, dtn.day, dtn.hour, dtn.minute, dtn.second))
reporter = open(reportfile, 'w')
orig_stdout = sys.stdout
sys.stdout = reporter

file_regex = re.compile(filePattern)

""" Stitch the reference round and channel """
if not stitchRef in rounds:
    raise ValueError("Stitching reference round is not in rounds list: {}".format(rounds))

""" Copy images to stitching folder"""
refImgPaths = []    # contains the path to images-to-be-stitched in each round
for fov in fovs:
    fov_files = os.listdir(os.path.join(data_dir, fov))

    for file in fov_files:
        mtch = file_regex.match(file)
        if mtch is not None:
            if (mtch.group('rndName') == stitchRef) and (mtch.group('ch') == stitchChRef):
                refImgPaths.append(os.path.join(data_dir, fov, file))

""" Making the reference tile configuration file """
intialRefTileConfigFile = "Ref_{0}_{1}_TileConfig.txt".format(stitchRef, stitchChRef)
createTileConfig([os.path.basename(file) for file in refImgPaths], 
                tileLocs,
                os.path.join(stitch_dir, intialRefTileConfigFile))

"""Move reference images to the stitch directory"""
copy2dir(refImgPaths, stitch_dir)

""" Stitch the reference channel with random-image fusion"""
print(datetime.now().strftime("%Y-%m-%d_%H:%M:%S: Stitching reference {0}, {1}".format(stitchRef, stitchChRef)))
refTileConfigFile = "Ref_{0}_{1}_TileConfig.txt".format(stitchRef, stitchChRef)
f_pat = re.sub(fov_pat, fov_sub, os.path.basename(refImgPaths[0])) # ImageJ sequence pattern

# refStitcher = IJS.IJ_Stitch(input_dir=stitch_dir, output_dir=stitch_dir, file_names=f_pat,
#                          imagej_path = ij_path, Type = 'Grid: row-by-row', Order = 'Left & Up', 
#                          tile_overlap = tileOverlap, grid_size_x=grid_size_x, grid_size_y=grid_size_y, 
#                          output_textfile_name=refTileConfigFile, 
#                          fusion_method = 'Intensity of random input tile',
#                          compute_overlap=True, macroName='{0}_{1}.ijm'.format(stitchRef, stitchChRef),
#                          output_name = 'Ref_{0}_{1}_random_fusion.tif'.format(stitchRef, stitchChRef))

refStitcher = IJS.IJ_Stitch(input_dir=stitch_dir, output_dir=stitch_dir, file_names=f_pat,
                           imagej_path = ij_path, Type = 'Positions from file', 
                           Order = 'Defined by TileConfiguration', 
                           layout_file = intialRefTileConfigFile,
                           output_textfile_name=refTileConfigFile, 
                           output_name = 'Ref_{0}_{1}_random_fusion.tif'.format(stitchRef, stitchChRef),
                           compute_overlap=True, macroName='{0}_{1}.ijm'.format(stitchRef, stitchChRef),
                           fusion_method = 'Intensity of random input tile')
res = refStitcher.run()
writeReport(res)

""" Stitch everything using the reference TileConfig"""
for rnd in rounds:
    """ Copy images to stitching folder"""
    thisRnd = {}    # contains the path to images-to-be-stitched in each round
    for fov in fovs:
        fov_files = os.listdir(os.path.join(data_dir, fov))

        for file in fov_files:
            mtch = file_regex.match(file)
            if mtch is not None:
                if mtch.group('rndName') == rnd:
                    add2dict2dict(mtch.group('ch'), os.path.join(data_dir, fov, file), thisRnd)
                    
    """Reorganize"""
    chans = list(thisRnd)
    for ch in chans:
        copy2dir(thisRnd[ch], stitch_dir)
#     print(thisRnd)
    
    """ Stitch the non-reference channels"""
    for nch in chans:
        nrefTileConfig = "{0}-to-{1}_{2}_TileConfig.registered.txt".format(stitchRef, rnd, nch)
        changeTileConfig(reffile=os.path.join(stitch_dir, "Ref_{0}_{1}_TileConfig.registered.txt".format(stitchRef, stitchChRef)),
                         nrefile=os.path.join(stitch_dir, nrefTileConfig), 
                         nrefNames=[os.path.basename(f) for f in thisRnd[nch]], 
                         fov_pat = fov_pat
                         )
        
        f_pat = re.sub(fov_pat, fov_sub, os.path.basename(thisRnd[nch][0])) # ImageJ sequence pattern
        
        print(datetime.now().strftime("%Y-%m-%d_%H:%M:%S: Stitching round {0}, {1} using the coordinates from {2}".format(rnd, nch, stitchRef)))
        nonRefStitcher = IJS.IJ_Stitch(input_dir=stitch_dir, output_dir=stitch_dir, file_names=f_pat,
                                       imagej_path = ij_path, Type = 'Positions from file', 
                                       Order = 'Defined by TileConfiguration', 
                                       layout_file = os.path.join(nrefTileConfig),
                                       compute_overlap=False, macroName='{0}_{1}.ijm'.format(rnd, nch),
                                       fusion_method = 'Max. Intensity')
        res = nonRefStitcher.run()
        writeReport(res)
#         break
    
    cleanUpImages(thisRnd, stitch_dir)
    reporter.flush()
#     break
   
    
""" Writing a CSV file for the coordinates of the registration reference cycle"""
allfiles = os.listdir(stitch_dir)
ref_ch = stitchChRef# if rnd in stitchChRefAlt else stitchChRef 
regRef_tileconfig_file = [f for f in allfiles 
                          if f == "Ref_{0}_{1}_TileConfig.registered.txt".format(stitchRef, stitchChRef)]
coords = readStitchInfo(os.path.join(stitch_dir, regRef_tileconfig_file[0]), filePattern[0:-1])
coords.to_csv(os.path.join(stitch_dir, 'registration_reference_coordinates.csv'), index = False)


sys.stdout = orig_stdout # restoring the stdout pipe to normal


""" Plotting the FOV map"""
allStitchedFiles = os.listdir(stitch_dir)
bgfile = [file for file in allStitchedFiles if (re.search(params['fovMap_bg'][0], file) is not None) and (re.search(params['fovMap_bg'][1], file) is not None) and (file.endswith(".tif"))]
if len(bgfile) > 1:
    raise Exception("Files {} were found for background of FOV map".format(bgfile))
bgImg = imread(os.path.join(stitch_dir, bgfile[0]))
npix, _, _ = getMetaData(metadataFile)

plotFOVMap(bgImg, coords_file=os.path.join(stitch_dir, 'registration_reference_coordinates.csv'), figure_height=12, 
    savefile=os.path.join(stitch_dir, "fov_map.pdf"), fov_size_px=npix)

