import pyvista as pv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import os

def get_total_iter(tmp_folder_list):
    foam_files_regex = fr'{tmp_folder_list}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    # print(len(foam_list))
    titles = []
    for i in range(len(foam_list)):
        title_iter = foam_list[i][-12:-4]
        title_iter = title_iter.lstrip('0')
        titles.append(title_iter)
    return titles[-1]

def find_nearest(array, value):

    array = np.asarray(array)
    ind = (np.abs(array - value)).argmin()

    return array[ind], ind

def calculate_quality_over_sim(data, geom_file, nx=1280, ny=550,skip=1):

    quality = np.array([])
    fracture = np.fromfile(geom_file, dtype=np.int8).reshape([ny, nx])[:,:550]
    for i in range(0, data.shape[-1], skip):
        foam_vof = data[:,:,i]

        foam_vof_calc = np.where(fracture==0, foam_vof, 100)

        fracture_voxels = len(np.where(foam_vof_calc < 100)[0])
        gas_voxels = len(np.where(foam_vof_calc==1)[0])
        # porosity = fracture_voxels/np.prod(fracture.shape)
        quality = np.append(quality, gas_voxels/fracture_voxels)

    return quality
