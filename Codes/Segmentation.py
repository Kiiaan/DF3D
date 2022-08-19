import sys, numpy as np, os
import cellpose.models as cp # cellpose has to be installed in the environment, and the models have to be downloaded in cellpose's favorite directory: ~/.cellpose/models
import pathlib
from urllib.parse import urlparse
from cellpose.core import parse_model_string

class Segmentor2D:
    """ Segments images of nuclei using cellpose"""
    def __init__(self, pretrained_model=None, flow_threshold=0.4):
        """
        pretrained_model: argument to CellposeModel, model name for built-in or full path to model for custom models
        """
        self.pretrained_model = pretrained_model
        if os.path.exists(pretrained_model): # it's a custom model
            print("Model at {}".format(self.pretrained_model))
            self.base_model = cp.CellposeModel(pretrained_model = self.pretrained_model)
        else: # it's a built-in model
            self.base_model = cp.CellposeModel(model_type=self.pretrained_model)

        self.flow_threshold = flow_threshold
        
    def segment_singleChannel(self, imgs, diameters = 40, out_files = None, **kwargs):
        """ Takes a list of nuclear stain images, an int or a list of average diameters 
            of nuclei in each image and runs Cellpose and returns masks in a list
            If diameter is None, run cellpose with automatic diameter detection and also returns estimated diameters
            Optional: out_files is a list of addresses to save the masks. 
        """
        print("Segmenting in single-channel mode with model {}.".format(self.pretrained_model))
        if len(imgs[0].shape) == 3:
            print("Images are 3D. TODO: 3D erosion")
            raise TypeError('3D image loaded instead of 2D')        
        
        masks = self.model_eval(imgs=imgs, channels = [0, 0], diams = diameters, **kwargs)
        
        if out_files is not None:
            self.save_masks(masks, out_files)
        return masks

    def segment_dualChannel(self, nuc_imgs, cyto_imgs, diameters = 40, out_files = None, **kwargs):
        """ Takes a list of nuclear stain images, an int or a list of average diameters 
            of nuclei in each image and runs Cellpose and returns masks in a list
            If diameter is None, run cellpose with automatic diameter detection and also returns estimated diameters
            Optional: out_files is a list of addresses to save the masks. 
        """
        print("Segmenting cells in dual-channel mode with model {}.".format(self.pretrained_model))
        if len(nuc_imgs[0].shape) == 3:
            print("Images are 3D. TODO: 3D erosion")
            raise TypeError('3D image loaded instead of 2D')        
        
        rgb_list = [np.stack([cyt, nuc, np.zeros_like(cyt)], axis=2) for cyt, nuc in zip(cyto_imgs, nuc_imgs)]
        masks=self.model_eval(imgs=rgb_list, channels = [1, 2], diams = diameters, **kwargs)
        
        if out_files is not None:
            self.save_masks(masks, out_files)
        return masks

    def model_eval(self, imgs, diams, channels=None, **kwargs):
        initial_masks = []
        if not isinstance(diams, list):
            diameters = len(imgs) * [diams]
        for i, img in enumerate(imgs):
            mask, _, _ = self.base_model.eval([img], channels = channels, diameter = diameters[i], flow_threshold=self.flow_threshold, **kwargs)
            if mask[0].max() < 2**16 - 1:
                initial_masks.append(mask[0].astype(np.uint16))
            else:
                initial_masks.append(mask[0])
        return initial_masks

    def save_masks(self, masks, files):
        for i, (file, mask) in enumerate(zip(files, masks)):
            print('Saving mask {0} in {1}.'.format(i, file))
            np.save(file, mask)
        
