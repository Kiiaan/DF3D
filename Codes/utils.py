import xml.etree.ElementTree as ET
import numpy as np 
import os

import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imshow
from matplotlib.patches import Rectangle


def getMetaData(metadataXml):
	""" parses a metadata .xml file and returns 
		1) size of FOVs in pixels
		2) physical dimension of the voxels
		3) #FOVs
    """ 
	tree = ET.parse(metadataXml)
	root = tree.getroot()
	dimDscrpt = [item for item in root.findall("./Image/ImageDescription/Dimensions/DimensionDescription")]
	# idDict = ['1' : 'X', '2' : 'Y', '3' : 'Z']
	n_pixels = {dimInfo.attrib['DimID']: int(dimInfo.attrib['NumberOfElements']) for dimInfo in dimDscrpt if dimInfo.attrib['DimID'] in ['1', '2', '3']}
	voxelSizes = {}
	for dimInfo in dimDscrpt: 
		if dimInfo.attrib['DimID'] not in ['1', '2', '3']:
			continue
		unit = dimInfo.attrib['Unit']
		if unit == 'um':
			scaleFac = 1
		elif unit == 'mm':
			scaleFac = 10 ** 3
		elif unit == 'm':
			scaleFac = 10 ** 6
		voxelSizes[dimInfo.attrib['DimID']] = scaleFac * float(dimInfo.attrib['Length']) / n_pixels[dimInfo.attrib['DimID']]


	tiles = [tile for tile in root.findall("./Image/Attachment/Tile")]

	return n_pixels, voxelSizes, len(tiles)


def getTileLocs(metadataXml):
    """ Getting the nominal location of the tiles. In the metadata file, the location unit is meters. 
        We convert this into pixels by finding the pixel size. The 0,0 coordinate will correspond to the first tile.
        For now, we assume the pixel size in x and y directions are similar and this value doesn't changed per cycle.
    """
    tree = ET.parse(metadataXml)
    root = tree.getroot()
    tiles = [tile for tile in root.findall("./Image/Attachment/Tile")]

    dimInfo = [item for item in root.findall("./Image/ImageDescription/Dimensions/DimensionDescription")][0]
    unit = dimInfo.attrib['Unit']
    if unit == 'um':
        scaleFac = 10 ** -6
    elif unit == 'mm':
        scaleFac = 10 ** -3
    elif unit == 'm':
        scaleFac = 1
    pxSize = scaleFac * float(dimInfo.attrib['Length']) / int(dimInfo.attrib['NumberOfElements'])

    x_0, y_0 = float(tiles[0].attrib['PosX']), float(tiles[0].attrib['PosY'])
    tiles_px = []
    for tile in tiles:
        pos_x = np.round((float(tile.attrib['PosX']) - x_0) / pxSize, 2)
        pos_y = np.round((float(tile.attrib['PosY']) - y_0) / pxSize, 2)
        tiles_px.append((pos_x, pos_y))
    return tiles_px

def getNumSections(metadataXml):
    """ Parses a metadata file to find the number of sections"""
    tree = ET.parse(metadataXml)
    root = tree.getroot()
    settings = [elem for elem in root.iter() if elem.tag == "ATLConfocalSettingDefinition"][0]
    return(int(settings.attrib['Sections']))

def plotFOVMap(bgImg, coords_file="registration_reference_coordinates.csv", figure_height=12, savefile="./fov_map.pdf", fov_size_px=1024):
    tile_coords = pd.read_csv(coords_file)
    tile_coords['x'] = (tile_coords['x'] - tile_coords['x'].min()).astype(int)
    tile_coords['y'] = (tile_coords['y'] - tile_coords['y'].min()).astype(int)

    fwidth = int(figure_height / bgImg.shape[0] * bgImg.shape[1])

    fig, ax = plt.subplots(figsize=(fwidth, figure_height))
    imshow(bgImg)

    for i, row in tile_coords.iterrows():
        plt.text(row['x'] + fov_size_px['1']/2, row['y'] + fov_size_px['2']/2, row['fov'], c = 'yellow',
                    horizontalalignment='center',
                    verticalalignment='center')
        rect = Rectangle((row['x'], row['y']), fov_size_px['1'], fov_size_px['2'], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    fig.savefig(savefile, dpi=200)

