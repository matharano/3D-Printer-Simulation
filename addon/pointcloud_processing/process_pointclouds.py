import numpy as np
import open3d as o3d
import os
import json
import copy
from tqdm import tqdm
from scipy import spatial

from . import generate_pointcloud as gp
from .PCL import PointCloud

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def merge_pointclouds(pcl1, pcl2, saving_path:str, calibration_filepath:str="data/calibration/calibration_files/transformation_parameters.json") -> np.ndarray:
    """Merge two pointclouds, usually obtained from the laser scanners, by applying transformations described by a calibration json.
    Pointclouds can be inputed as open3d pointclouds, numpy arrays or paths to those formats (.ply or .npy)."""

    ## import the calibration parameters
    with open(calibration_filepath, "r") as read_file:
        calib_params = json.load(read_file)
        transformation_matrix_icp = np.asarray(calib_params["transformation_matrix"])
        delta1_z = calib_params['delta1_z']
        delta2_z = calib_params['delta2_z']
        theta1_xz = calib_params['theta1_xz']
        theta1_yz = calib_params['theta1_yz']
        theta2_xz = calib_params['theta2_xz']
        theta2_yz = calib_params['theta2_yz']

    ## Import the to be calibrated parts and converting them into an numpy array in order to process them
    if type(pcl1) != type(pcl2):
        raise NameError('Given point clouds are in different types. Try again with a same type.')
    if type(pcl1) == str:   
        # if not os.path.isdir(pcl1) or not os.path.isdir(pcl2):
        #     raise NameError('Given point clouds do not exist.')
        extension = pcl1.split('.')[-1].lower()
        if extension == 'ply':
            pcl1 = o3d.io.read_point_cloud(pcl1)
            pcl2 = o3d.io.read_point_cloud(pcl2)
        elif extension == 'npy':
            pcl1 = np.load(pcl1)
            pcl2 = np.load(pcl2)
    if type(pcl1) == o3d.geometry.PointCloud:
        pcl1 = np.asarray(pcl1.points)
        pcl2 = np.asarray(pcl2.points)
    if type(pcl1) != np.ndarray or type(pcl2) != np.ndarray:
        raise NameError('Files could not be properly opened.')
    
    # ## Inverting the Z-values making it easier to process them
    pcl1[:,2] = pcl1[:,2]*-1
    pcl2[:,2] = pcl2[:,2]*-1

    ## Creating PointCloud classes (self made class from PCL.py)
    pcl1 = PointCloud(pcl1)
    pcl2 = PointCloud(pcl2)

    ## Creating an Z-area where points are kept (removing outliers)
    ##pcl1.clean('z', -1000, -500)
    ##pcl2.clean('z', -1000, -500)

    ## Calibrating with the calibration parameters
    ## Normalize to the minimum in z direction
    pcl1.shift('z', delta1_z)
    pcl2.shift('z', delta2_z)
    ## Correct the tilt in z direction
    pcl1.rotate('x', 'z', theta1_xz)
    pcl1.rotate('y', 'z', theta1_yz)
    pcl2.rotate('x', 'z', theta2_xz)
    pcl2.rotate('y', 'z', theta2_yz)

    # pcl1.clean('z', 0.3, 300)
    # pcl2.clean('z', 0.3, 300)
    
    ## Creating again numpy arrays in order to create PointCloud classes (this time we use the class from open3d)
    pcl1 = pcl1.transform_to_np()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcl1)

    pcl2 = pcl2.transform_to_np()
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcl2)

    ## visualization of the results
    ##draw_registration_result(pcd1, pcd2, transformation_matrix_icp)

    ## transforming npy3 into the right coordinate system using the generated transformation
    npy1_transformed = np.asarray(pcd1.transform(transformation_matrix_icp).points)

    ## merging of both pointclouds
    merged_pcl = np.append(npy1_transformed, pcl2, axis=0)
    
    ## saving the merged pcl as .npy
    ##gp.save_as_npArray(Data=merged_pcl, name=str(layer), LLS="_merged", path = "/data/Numpy_Arrays/merged/")
    
    ## saving the merged pcl as .ply
    if len(saving_path) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pcl)
        if saving_path[-4:] != '.ply':
            saving_path += '.ply'
        o3d.io.write_point_cloud(saving_path, pcd)
        
    return merged_pcl

def layerize_pointclouds(layers, distance_threshold:float=0.2, workers:int=-1):
    """Given a list of point clouds, sorted from bottom to top, iterates so that each layer represents only the added points."""
    diff_layers = [layers[0]]
    for i in tqdm(range(1, len(layers))):
        basis = spatial.KDTree(layers[i-1])
        raw_addition = layers[i]
        dists, idx = basis.query(raw_addition, 1, p=2, distance_upper_bound=distance_threshold, workers=workers)
        addition = raw_addition[dists == np.inf]
        diff_layers.append(addition)
    return diff_layers

if __name__ == '__main__':
    folder = 'data/pointclouds/raw'
    calibration_path = 'data/calibration/calibration_files/transformation_parameters.json'
    merged_layers = []
    for i in range(20):
        saving_path = 'data/pointclouds/layerize/%d_merged.ply' % i
        merged_pcl = merge_pointclouds(
            os.path.join(folder, '%d_LLS1.ply' % i),
            os.path.join(folder, '%d_LLS2.ply' % i),
            saving_path,
            calibration_path
        )
        merged_layers.append(merged_pcl)
    layers = layerize_pointclouds(merged_layers)
    for (i, layer) in enumerate(layers):
        gp.generate_pcl(layer, path="data/pointclouds/layerize/piece/layerized/", name = "", LLS =str(i))