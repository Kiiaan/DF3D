#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kian: added 201011 after having trouble importing SimpleITK normally
import sys
sitkPath = '/media/Home_Raid1_Voyager/kian/packages/201011_SimpleElastix/build/SimpleITK-build/Wrapping/Python'
sys.path.insert(1, sitkPath)
import SimpleITK as sitk
import tifffile
import numpy as np
import os, re
from os.path import join as pathjoin
from datetime import datetime


class ThreeDimensionalAligner():
    def __init__(self, movingImgFiles, refImgFiles, transform = "affine", NumberOfResolutions = 5, MaximumNumberOfIterations = 1000, 
                    NumberOfSpatialSamples = 4000, transParamFile = None, voxelSize = (0.0001, 0.0001, 0.001)):
        self.movingImgFiles = movingImgFiles
        self.refImgFiles = refImgFiles
        self.transform = transform
        self.NumberOfResolutions = NumberOfResolutions
        self.MaximumNumberOfIterations = MaximumNumberOfIterations
        self.NumberOfSpatialSamples = NumberOfSpatialSamples
        self.transParamFile = transParamFile
        self.voxelSize = voxelSize
        
    def findTransformParameters(self):   
        """ running elastix on moving and ref images to find the transform parameter map between them """
        self.elastixImageFilter = sitk.ElastixImageFilter() # The basic object to do the transformation
        
        """ Setting the transformation parameters"""        
        parameterMap = self.elastixImageFilter.GetDefaultParameterMap(self.transform) # getting the dafault parameter map for our transformation of interest
        parameterMap['NumberOfHistogramBins'] = ['64'] # a parameter for the image comparison metric, AdvancedMattesMutualInformation, that we are using.
        parameterMap['MaximumNumberOfIterations'] = [str(self.MaximumNumberOfIterations)] # number of iterations per aligning each resolution
        parameterMap['NumberOfResolutions'] = [str(self.NumberOfResolutions)] # number of resolution-decreasing alignments. This is the most critical parameter
        parameterMap['NumberOfSpatialSamples'] = [str(self.NumberOfSpatialSamples)] # number of random samples drawn for image comparison during optimization
        parameterMap['WriteIterationInfo'] = ['true'] # This command writes the report in the current working directory, so we have to move the files later    
        parameterMap['AutomaticScalesEstimation'] = ["false"]
        parameterMap['Scales'] = ['2']  # the rotations we expect (in radians) are almost in the same order of magnitude as the translations we expect (in mm)

        
        self.elastixImageFilter.SetParameterMap(parameterMap) # setting the parameter map to our transformation object
        self.elastixImageFilter.SetMovingImage(self.readMovingImage()) # Setting the origin image, the one we want to transform
        self.elastixImageFilter.SetFixedImage(self.readRefImage()) # Setting the destination/final image
        
        self.elastixImageFilter.LogToFileOn()
        self.elastixImageFilter.SetOutputDirectory("./")

        self.elastixImageFilter.Execute()   # running the transformation
        self.transformParameterMap = self.elastixImageFilter.GetTransformParameterMap() # saving the optimized transformation parameters
        
        if not self.transParamFile is None:
            self.writeParameterFile(reportName = self.transParamFile) # write the parameter map to folder MetaData

    def applyTransformation(self, movingImgFiles_list, outImgFiles_list):
        """ Applies the computed transform to all the moving images within movingImgFiles_list. 
            movingImgFiles_list is a list of lists, each inner list contains one stack.
            outImgFiles_list is a list of lists, each inner list specifies the path to the transformed images.
        """
        self.transformixImageFilter = sitk.TransformixImageFilter() 
        self.transformParameterMap = self.getTransformParameterMap()
        self.transformParameterMap[0]['FinalBSplineInterpolationOrder'] = ['1']
        self.transformixImageFilter.SetTransformParameterMap(self.transformParameterMap) 
        
        for movingImgFiles, outImgFiles in zip(movingImgFiles_list, outImgFiles_list):
            images3D_input = self.readImage(movingImgFiles)
            self.transformixImageFilter.SetMovingImage(images3D_input)
            self.transformixImageFilter.Execute()

            outputImage = self.transformixImageFilter.GetResultImage()

            self.writeImage(sitk.Cast(outputImage, sitk.sitkUInt8), outImgFiles)
            # sitk.WriteImage(sitk.Cast(outputImage, sitk.sitkUInt8),
            #                 outImgFiles)
        
    
    def readMovingImage(self):
        im = sitk.ReadImage(self.movingImgFiles)
        im.SetSpacing(self.voxelSize)
        return(im)
    
    def readRefImage(self):
        im = sitk.ReadImage(self.refImgFiles)
        im.SetSpacing(self.voxelSize)
        return(im)

    def readImage(self, path):
        im = sitk.ReadImage(path)
        im.SetSpacing(self.voxelSize)
        return(im)        
    
    def writeParameterFile(self, reportName):
        self.elastixImageFilter.WriteParameterFile(parameterMap = self.transformParameterMap[0], filename = reportName)
     
    def getTransformParameterMap(self):
        return self.transformParameterMap

    def writeImage(self, img_sitk, outfiles):
        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.uint8)
        for im, out in zip(img_np, outfiles):
            # print('X', out)
            tifffile.imwrite(out, im, imagej=True, photometric = 'minisblack', compression="zlib") # "LZW")

