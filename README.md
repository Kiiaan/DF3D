# DART-FISH Pipeline
This repo contains the necessary codes to process DART-FISH imaging data (https://doi.org/10.1101/2023.08.16.553610). It processes the raw fluorescent images to obtain various outputs including the count matrix, segmentation mask and spot table. For a demonstration of the pipeline, see https://zenodo.org/record/8253772.
## How to run
1. All the parameters relevant to the various steps of the pipeline are gathered in DF3D/Codes/params.yaml
* This includes the path to the raw data, and the paths to all the subsequent directories.
2. The modules of the code, i.e., 3D registration, 2D projection, stitching, decoding, segementation, are run sequentially by DF3D/Codes/main.py with params.yaml as input
```
# working directory should be set at DF3D/Codes/
python main.py params.yaml
# OR
bash run_pipeline.sh
```
Note that the most computation-intensive steps are 3D image registration and sparse deconvolution (SpD). Each field of view takes 5 minutes to register and 10 minutes to decode on our server (with Intel Xeon CPU E5-2697A v4 @ 2.60GHz). But the exact run time of the pipeline depends on the number of fields of view, the number of z-stacks, and the number of cores used for each step. 

## Raw data format
Those modules that import the microscope raw data, e.g., image registration (AlignerDriver_3D.py), were written for the directory structure provided by a Leica SP8 microscope:
```
"project_directory"/
  "round_name1"/
    "round_name1"_s*_z*_ch00.tif
    "round_name1"_s*_z*_ch01.tif
    "round_name1"_s*_z*_ch02.tif
    "round_name1"_s*_z*_ch03.tif
    ...
  "round_name2"/
    ...
  ...  
```
where `s` is the field of view (FOV) number and  ```z``` the number for z-stack. The related files can be modified to adapt any directory structure without the need to change other modules. 

## Installation procedure:
1. Install a conda environment using the provided `.yaml` file
2. Build and install SimpleElastix
3. (Uncertain) Resolve errors regarding `GLIBCXX_3.4.26`
Note that this pipeline has been tested only on Ubuntu 22.04.1.
### Installing the conda environment
Installing the environment requires having Anaconda. See [here]([https://www.anaconda.com](https://docs.anaconda.com/free/anaconda/install/linux/) if you don't have Anaconda installed. The installation should not take more than 15 minutes. 
```
conda env create -f DF221115_env.yaml
conda activate DF_221115
```
### Installing SimpleElastix from scratch:
This step can take up to a few hours depending on the system. 
```
cd ~/packages
mkdir SimpleElastix_221115
cd SimpleElastix_221115/
git clone https://github.com/SuperElastix/SimpleElastix
mkdir SE_build
cd SE_build
cmake ../SimpleElastix/SuperBuild
make -j8
cd SimpleITK-build/Wrapping/Python/
python Packaging/setup.py install --user
```
### Error with `GLIBCXX_3.4.26`
I received such error with this environment's settings when running the segmentation_driver.py:
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /media/Home_Raid1_Voyager/kian/anaconda3/envs/DF_221115/lib/python3.8/site-packages/scipy/linalg/_matfuncs_sqrtm_triu.cpython-38-x86_64-linux-gnu.so)
```
I couldn't figure out why this is happening but [this StackOverflow thread](https://stackoverflow.com/questions/54948216/usr-lib-x86-64-linux-gnu-libstdc-so-6-version-glibcxx-3-4-21-not-found-req) seems to have solutions for it. What I tried and worked is this:
1. Add this to the `~/.bashrc` file: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/DF_221115/lib`
2. Run `source ~/.bashrc`

