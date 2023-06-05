import functools
import multiprocessing as mp
import os
from functools import partial
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd
import SimpleITK as sitk

targetDir= '/workspaces/jax_cpu_experiments_b/explore/picai_unpacked'
def unpackk(zipDir,targetDir):
    with ZipFile(zipDir, "r") as zip_ref:
        for name in zip_ref.namelist():
            #ignoring all corrupt files
            try:
                zip_ref.extract(name, targetDir)
            except BadZipFile as e:
                print(e)
    

unpackk( '/workspaces/jax_cpu_experiments_b/explore/picai/picai_public_images_fold0.zip', targetDir)      
unpackk( '/workspaces/jax_cpu_experiments_b/explore/picai/picai_public_images_fold1.zip', targetDir)      
unpackk( '/workspaces/jax_cpu_experiments_b/explore/picai/picai_public_images_fold2.zip', targetDir)      
unpackk( '/workspaces/jax_cpu_experiments_b/explore/picai/picai_public_images_fold3.zip', targetDir)      
unpackk( '/workspaces/jax_cpu_experiments_b/explore/picai/picai_public_images_fold4.zip', targetDir)      

