# Installation instructions
conda env create -f DF221115_env.yaml

conda activate DF_221115

install SimpleElastix from scratch:

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

#packages in environment at /media/Home_Raid1_Voyager/kian/anaconda3/envs/DF_220412:

-# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
argon2-cffi               20.1.0                   pypi_0    pypi
async-generator           1.10                     pypi_0    pypi
attrs                     20.2.0                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    3.2.1                    pypi_0    pypi
boto3                     1.15.16                  pypi_0    pypi
botocore                  1.18.16                  pypi_0    pypi
bz2file                   0.98                       py_0    conda-forge
ca-certificates           2021.10.8            ha878542_0    conda-forge
cachetools                4.1.1                    pypi_0    pypi
cellpose                  2.0.4                    pypi_0    pypi
certifi                   2021.10.8        py37h89c1867_1    conda-forge
cffi                      1.14.3                   pypi_0    pypi
chardet                   3.0.4                    pypi_0    pypi
click                     7.1.2                    pypi_0    pypi
cutadapt                  1.18             py37h14c3975_1    bioconda
cycler                    0.10.0                   py37_0  
dataclasses               0.6                      pypi_0    pypi
dbus                      1.13.16              hb2f20db_0  
decorator                 4.4.2                    pypi_0    pypi
defusedxml                0.6.0                    pypi_0    pypi
diskcache                 5.0.3                    pypi_0    pypi
entrypoints               0.3                      pypi_0    pypi
expat                     2.2.10               he6710b0_2  
fastremap                 1.12.2                   pypi_0    pypi
fontconfig                2.13.0               h9420a91_0  
freetype                  2.10.3               h5ab3b9f_0  
git                       2.23.0          pl526hacde149_0    anaconda
glib                      2.65.0               h3eb4bd4_0  
google-api-core           1.22.4                   pypi_0    pypi
google-auth               1.22.1                   pypi_0    pypi
google-cloud-core         1.4.3                    pypi_0    pypi
google-cloud-storage      1.31.2                   pypi_0    pypi
google-crc32c             1.0.0                    pypi_0    pypi
google-resumable-media    1.1.0                    pypi_0    pypi
googleapis-common-protos  1.52.0                   pypi_0    pypi
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb31296c_0  
h5py                      2.10.0                   pypi_0    pypi
icu                       58.2                 he6710b0_3  
idna                      2.10                     pypi_0    pypi
imageio                   2.9.0                    pypi_0    pypi
imglyb                    0.4.0            py37h5ca1d4c_0    conda-forge
importlib-metadata        2.0.0                    pypi_0    pypi
intel-openmp              2020.2                      254  
ipywidgets                7.5.1                    pypi_0    pypi
jgo                       0.5.0            py37hc8dfbb8_1    conda-forge
jinja2                    2.11.2                   pypi_0    pypi
jmespath                  0.10.0                   pypi_0    pypi
joblib                    0.17.0                   pypi_0    pypi
jpeg                      9b                   h024ee3a_2  
jsonschema                3.2.0                    pypi_0    pypi
jupyter                   1.0.0                    pypi_0    pypi
jupyter-console           6.2.0                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
kiwisolver                1.2.0            py37hfd86e86_0  
krb5                      1.18.2               h173b8e3_0    anaconda
lcms2                     2.11                 h396b838_0  
ld_impl_linux-64          2.33.1               h53a641e_7  
libcurl                   7.71.1               h20c2e04_1    anaconda
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libllvm10                 10.0.1               hbcb73fb_5  
libpng                    1.6.37               hbc83047_0  
libssh2                   1.9.0                h1ba5d50_1    anaconda
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h2733197_1  
libuuid                   1.0.3                h1bed415_2  
libxcb                    1.14                 h7b6447c_0  
libxml2                   2.9.10               he19cac6_1  
llvmlite                  0.34.0           py37h269e1b5_4  
lz4-c                     1.9.2                heb0550a_3  
markupsafe                1.1.1                    pypi_0    pypi
matplotlib                3.3.1                         0  
matplotlib-base           3.3.1            py37h817c723_0  
maven                     3.6.0                         0  
mistune                   0.8.4                    pypi_0    pypi
mkl                       2020.2                      256  
mkl-service               2.3.0            py37he904b0f_0  
mkl_fft                   1.2.0            py37h23d657b_0  
mkl_random                1.1.1            py37h0573a6f_0  
mpmath                    1.1.0                    pypi_0    pypi
mxnet-mkl                 1.6.0                    pypi_0    pypi
natsort                   7.0.1                    pypi_0    pypi
nbclient                  0.5.0                    pypi_0    pypi
nbconvert                 6.0.7                    pypi_0    pypi
nbformat                  5.0.7                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1  
nest-asyncio              1.4.1                    pypi_0    pypi
networkx                  2.5                      pypi_0    pypi
notebook                  6.1.4                    pypi_0    pypi
numba                     0.51.2           py37h04863e7_1  
numpy                     1.21.6                   pypi_0    pypi
olefile                   0.46                     py37_0  
opencv-python-headless    4.4.0.44                 pypi_0    pypi
openjdk                   8.0.152              h7b6447c_3  
openssl                   1.1.1h               h7b6447c_0    anaconda
packaging                 20.4                     pypi_0    pypi
pandas                    1.1.3                    pypi_0    pypi
pandocfilters             1.4.2                    pypi_0    pypi
pcre                      8.44                 he6710b0_0  
perl                      5.26.2               h14c3975_0    anaconda
pigz                      2.6                  h27cfd23_0  
pillow                    7.2.0            py37hb39fc2d_0  
pip                       20.2.3                   py37_0  
prometheus-client         0.8.0                    pypi_0    pypi
protobuf                  3.13.0                   pypi_0    pypi
psutil                    5.7.2            py37h7b6447c_0  
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.20                     pypi_0    pypi
pyimagej                  0.5.0                    py37_0    conda-forge
pyjnius                   1.1.3           py37hf484d3e_1001    conda-forge
pyparsing                 2.4.7                      py_0  
pyqt                      5.9.2            py37h05f1152_2  
pyqtgraph                 0.11.0rc0                pypi_0    pypi
pyrsistent                0.17.3                   pypi_0    pypi
python                    3.7.9                h7579374_0  
python-dateutil           2.8.0                    pypi_0    pypi
python-graphviz           0.8.4                    pypi_0    pypi
python_abi                3.7                     1_cp37m    conda-forge
pytz                      2020.1                   pypi_0    pypi
pywavelets                1.1.1                    pypi_0    pypi
pyyaml                    5.3.1                    pypi_0    pypi
qt                        5.9.7                h5867ecd_1  
qtconsole                 4.7.7                    pypi_0    pypi
qtpy                      1.9.0                    pypi_0    pypi
read-roi                  1.6.0                    pypi_0    pypi
readline                  8.0                  h7b6447c_0  
regional                  1.1.2                    pypi_0    pypi
requests                  2.24.0                   pypi_0    pypi
rsa                       4.6                      pypi_0    pypi
s3transfer                0.3.3                    pypi_0    pypi
scikit-image              0.15.0                   pypi_0    pypi
scikit-learn              0.23.2                   pypi_0    pypi
scipy                     1.5.2            py37h0b6359f_0  
scyjava                   0.4.0                    py37_0    conda-forge
semantic-version          2.8.5                    pypi_0    pypi
send2trash                1.5.0                    pypi_0    pypi
setuptools                50.3.0           py37hb0f4dca_1  
showit                    1.1.4                    pypi_0    pypi
sip                       4.19.8           py37hf484d3e_0  
six                       1.15.0                     py_0  
slicedimage               4.1.1                    pypi_0    pypi
sqlite                    3.33.0               h62c20be_0  
starfish                  0.2.1                    pypi_0    pypi
sympy                     1.6.2                    pypi_0    pypi
tbb                       2020.3               hfd86e86_0  
terminado                 0.9.1                    pypi_0    pypi
testpath                  0.4.4                    pypi_0    pypi
threadpoolctl             2.1.0                    pypi_0    pypi
tifffile                  2020.10.1                pypi_0    pypi
tk                        8.6.10               hbc83047_0  
torch                     1.11.0                   pypi_0    pypi
tornado                   6.0.4            py37h7b6447c_1  
tqdm                      4.50.2                   pypi_0    pypi
trackpy                   0.4.2                    pypi_0    pypi
typing-extensions         4.1.1                    pypi_0    pypi
urllib3                   1.25.10                  pypi_0    pypi
validators                0.18.1                   pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
wheel                     0.35.1                     py_0  
widgetsnbextension        3.5.1                    pypi_0    pypi
xarray                    0.16.1                   pypi_0    pypi
xopen                     0.7.3                      py_0    bioconda
xz                        5.2.5                h7b6447c_0  
zipp                      3.3.0                    pypi_0    pypi
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.5                h9ceee32_0  
