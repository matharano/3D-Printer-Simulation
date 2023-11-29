# -*- coding: utf-8 -*-
"""
@author: grh

Project: SmoPa3D

Description: This code is used to calibrate the laser light section sensors. 
For this purpose, a standardized component must be placed on the build platform, which is scanned.
Based on the captured data, calibration data is generated, which must be applied to all captured data.
"""
import os
import json
import logging
import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from json import JSONEncoder
from .PCL import PointCloud, Calibration

from . import generate_pointcloud as gp

logger = logging.getLogger('SmoPa3D')  # Logger should be defined at demonstrator-orchestration

#===============================================================================
## Helper functions
#=============================================================================== 
## Look up "global registration open3d" http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
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

## Look up "global registration open3d" http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def preprocess_registration(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

## Look up "global registration open3d" http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    ##source.transform(trans_init)
    ##draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_registration(source, voxel_size)
    target_down, target_fpfh = preprocess_registration(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

## Look up "global registration open3d" http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,  False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

## Function for displaying in- and outliers
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
## Function for removal of outliers
def remove_outlier(cloud):
    ## Downsample the point cloud with a voxel of 0.02
    voxel_down_pcl = cloud.voxel_down_sample(voxel_size=0.02)
    ## Statistical oulier removal
    cloud, ind = voxel_down_pcl.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ##display_inlier_outlier(voxel_down_pcl, ind)
    return cloud

def preprocess_pointcloud(pcl):
    ## Converting of the pcl into an numpy array in order to process them
    npy = np.asarray(pcl.points)

    ## Inverting the Z-values making it easier to process them
    npy[:,2] = npy[:,2]*-1

    ## Creating PointCloud classes (self made class from PCL.py)
    pcl = PointCloud(npy)

    
    ## Creating an X,Y and Z-area where only the relevant points are kept
    pcl.clean('z', -1000, 300)

    return pcl

## Definition of class in order to encode np arrays to json data
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#===============================================================================
## Creation of calibration parameters using three different Pointclouds
## 1. Surface of building platform
## 2. Surface of printed part on building platform
## 3. Features in order to calculate a transformation
#=============================================================================== 
def get_calibration_parameters(
        calib_plate_1 = os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements", "plate_LLS1.ply"),
        calib_plate_2=os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements","plate_LLS2.ply"),
        surface_layer_1=os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements","part_surface_LLS1.ply"),
        surface_layer_2=os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements","part_surface_LLS2.ply"),
        last_layer_1=os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements","last_layer_LLS1.ply"),
        last_layer_2=os.path.join(os.getcwd(), "data", "calibration", "calibration_measurements","last_layer_LLS2.ply"),
        parameters_path:str=(os.getcwd() + "/data/calibration/calibration_files/transformation_parameters.json"),
        view_steps:bool=logger.level <= 20
        ):

    #===============================================================================
    ## Calibration of building plate
    #===============================================================================
    ## Import the calibration pcls
    pcl_1 = o3d.io.read_point_cloud(calib_plate_1)
    pcl_2 = o3d.io.read_point_cloud(calib_plate_2)

    ## Removing outliers
    pcl_1 = remove_outlier(pcl_1)
    pcl_2 = remove_outlier(pcl_2)

    ## Preprocessing of Pointclouds
    pcl1 = preprocess_pointcloud(pcl_1)
    pcl2 = preprocess_pointcloud(pcl_2)
    
    ## Creating a Calibration class (self made class from PCL.py)
    calibration = Calibration(pcl1, pcl2)

    ## Normalize to the minimum in z direction
    calibration.normalize_to_minimum('z')

    ## Correct the tilt in z direction
    calibration.z_leveling()

    ## Shift both point clouds to bed level. This should be where the highest point density is found
    calibration.bed_leveling()

    if view_steps: plt.show()
    
    ## Setting the zero plane by taking the top 99% quantile and shifting the zero point of the z-axis to that value
    data_1 = np.reshape(pcl1._data, (-1, 3))
    quant_1 = np.quantile(data_1[:,2], 0.99)
    data_1[:,2] = data_1[:,2] - quant_1

    data_2 = np.reshape(pcl2._data, (-1, 3))
    quant_2 = np.quantile(data_2[:,2], 0.99)
    data_2[:,2] = data_2[:,2] - quant_2

    """
    ## Visualization
    plt.plot(data_1[:,1], data_1[:,2], 'o', label='updated data 1')
    plt.legend()
    plt.show()
    plt.plot(data_2[:,1], data_2[:,2], 'o', label='updated data 2')
    plt.legend()
    plt.show()
    """
    
    ## Getting the calibration parameters
    plate_delta1_z = float(calibration.calibration_parameters['delta1_z']) + quant_1
    plate_delta2_z = float(calibration.calibration_parameters['delta2_z']) + quant_2
    plate_theta1_xz = float(calibration.calibration_parameters['theta1_xz'])
    plate_theta1_yz = float(calibration.calibration_parameters['theta1_yz'])
    plate_theta2_xz = float(calibration.calibration_parameters['theta2_xz'])
    plate_theta2_yz = float(calibration.calibration_parameters['theta2_yz'])


    logger.info("Both LLS are calibrated regarding the building plate")
    #===============================================================================
    ## Calibration of even surface of printed calibration parts
    #===============================================================================
    ## Import the calibration pcls
    pcl_1 = o3d.io.read_point_cloud(surface_layer_1)
    pcl_2 = o3d.io.read_point_cloud(surface_layer_2)

    ## Removing outliers
    pcl_1 = remove_outlier(pcl_1)
    pcl_2 = remove_outlier(pcl_2)

    ## Preprocessing of Pointclouds
    pcl1 = preprocess_pointcloud(pcl_1)
    pcl1.clean('y', 50, 90)
    pcl1.clean('x', -30, -5)
    pcl2 = preprocess_pointcloud(pcl_2)
    pcl2.clean('y', 10, 50)
    pcl2.clean('x', 5, 30)

    ## Calibrating with the previous calibration parameters
    ## Normalize to the minimum in z direction
    pcl1.shift('z', plate_delta1_z)
    pcl2.shift('z', plate_delta2_z)
    ## Correct the tilt in z direction
    pcl1.rotate('x', 'z', plate_theta1_xz)
    pcl1.rotate('y', 'z', plate_theta1_yz)
    pcl2.rotate('x', 'z', plate_theta2_xz)
    pcl2.rotate('y', 'z', plate_theta2_yz)
    ## Cutting out the relevant area
    pcl1.clean('z', 0, 10)
    pcl2.clean('z', 0, 10)

    ## Creating a Calibration class (self made class from PCL.py)
    surface_calibration = Calibration(pcl1, pcl2)

    ## Correct the tilt in z direction
    surface_calibration.z_leveling()
    
    
    ## Visualization
    data_1 = np.reshape(pcl1._data, (-1, 3))
    data_2 = np.reshape(pcl2._data, (-1, 3))

    if view_steps:
        plt.show()
        plt.plot(data_1[:,0], data_1[:,2], 'o', label='xz updated data 1')
        plt.legend()
        plt.show()
        plt.plot(data_1[:,1], data_1[:,2], 'o', label='yz updated data 1')
        plt.legend()
        plt.show()
        plt.plot(data_2[:,0], data_2[:,2], 'o', label='xz updated data 2')
        plt.legend()
        plt.show()
        plt.plot(data_2[:,1], data_2[:,2], 'o', label='yz updated data 2')
        plt.legend()
        plt.show()
    

    ## Adapting the previous calibration parameters
    ##delta1_z = plate_delta1_z + float(surface_calibration.calibration_parameters['delta1_z'])
    delta1_z = plate_delta1_z
    ##delta2_z = plate_delta2_z + float(surface_calibration.calibration_parameters['delta2_z'])
    delta2_z = plate_delta2_z
    theta1_xz = plate_theta1_xz + float(surface_calibration.calibration_parameters['theta1_xz'])
    theta1_yz = plate_theta1_yz + float(surface_calibration.calibration_parameters['theta1_yz'])
    theta2_xz = plate_theta2_xz + float(surface_calibration.calibration_parameters['theta2_xz'])
    theta2_yz = plate_theta2_yz + float(surface_calibration.calibration_parameters['theta2_yz'])

    
    #===============================================================================
    ## Test if the calibration worked
    #===============================================================================
    pcl_1 = o3d.io.read_point_cloud(surface_layer_1)
    pcl_2 = o3d.io.read_point_cloud(surface_layer_2)

    ## Removing outliers
    pcl_1 = remove_outlier(pcl_1)
    pcl_2 = remove_outlier(pcl_2)

    ## Preprocessing of Pointclouds
    pcl1 = preprocess_pointcloud(pcl_1)
    pcl2 = preprocess_pointcloud(pcl_2)
    
    ## Calibrating with the previous calibration parameters
    ## Normalize to the minimum in z direction
    pcl1.shift('z', delta1_z)
    pcl2.shift('z', delta2_z)
    ## Correct the tilt in z direction
    pcl1.rotate('x', 'z', theta1_xz)
    pcl1.rotate('y', 'z', theta1_yz)
    pcl2.rotate('x', 'z', theta2_xz)
    pcl2.rotate('y', 'z', theta2_yz)

    pcl1.clean('z', -10, 10)
    pcl2.clean('z', -10, 10)

    data_1 = np.reshape(pcl1._data, (-1, 3))
    data_2 = np.reshape(pcl2._data, (-1, 3))
    
    ## Visualization
    if view_steps:
        plt.plot(data_1[:,0], data_1[:,2], 'o', label='updated data 1')
        plt.legend()
        plt.show()
        plt.plot(data_2[:,0], data_2[:,2], 'o', label='updated data 2')
        plt.legend()
        plt.show()
    
    logger.info("Finished calibration of the even surface")
    #===============================================================================
    ## Calibration of features of printed calibration parts and merging of both pointclouds
    #===============================================================================
    
    ## Import the calibration pcls
    pcl_1 = o3d.io.read_point_cloud(last_layer_1)
    pcl_2 = o3d.io.read_point_cloud(last_layer_2)

    ## Preprocessing of Pointclouds
    pcl1 = preprocess_pointcloud(pcl_1)
    pcl2 = preprocess_pointcloud(pcl_2)

    ## Calibrating with the previous calibration parameters
    ## Normalize to the minimum in z direction
    pcl1.shift('z', delta1_z)
    pcl2.shift('z', delta2_z)
    ## Correct the tilt in z direction
    pcl1.rotate('x', 'z', theta1_xz)
    pcl1.rotate('y', 'z', theta1_yz)
    pcl2.rotate('x', 'z', theta2_xz)
    pcl2.rotate('y', 'z', theta2_yz)
    ## Cutting out the relevant area
    pcl1.clean('z', 1.1, 10)
    pcl1.clean('y', 50, 90)
    pcl1.clean('x', -30, -5)

    pcl2.clean('z', 1.1, 10)
    pcl2.clean('y', 10, 50)
    pcl2.clean('x', 5, 30)

    ## Creating again numpy arrays in order to create PointCloud classes (this time we use the class from open3d)
    npy1 = pcl1.transform_to_np()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(npy1)

    npy2 = pcl2.transform_to_np()
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(npy2)

    ## Transfer of point clouds for processing in the registration algorithms
    target = pcd2
    source = pcd1

    ## Setting up of variables for the registration
    voxel_size = 0.5
    threshold = voxel_size * 0.4

    ## extracting geometric features from the original pointclouds
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source= source, target=target)

    ## Using the RANSAC for global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    ## Local registration using the icp-algorithm -> for more: http://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

    ## vizualisation of the results
    if view_steps: draw_registration_result(source, target, result_icp.transformation)

    logger.info(result_icp.fitness)
    logger.info(result_icp.inlier_rmse)

    ## Initalisation of dict for saving of JSON file
    data = {
        "delta1_z" : float(delta1_z),
        "delta2_z" : float(delta2_z),
        "theta1_xz" : float(theta1_xz),
        "theta1_yz" : float(theta1_yz),
        "theta2_xz" : float(theta2_xz),
        "theta2_yz" : float(theta2_yz),
        "fitness" : result_icp.fitness,
        "inlier_rmse" : result_icp.inlier_rmse,
        "transformation_matrix" : result_icp.transformation
    }

    # Serializing json
    json_object = json.dumps(data, cls=NumpyArrayEncoder, indent=10)

    # Writing to sample.json
    path = os.path.dirname(parameters_path)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(parameters_path, "w") as outfile:
        outfile.write(json_object)

    return calibration, result_icp

def calibrate_pcls(Layerheight, pcl1 =None, pcl2=None, calib_path = "/data/calibration/calibration_files/transformation_parameters.json", filetype = "NumpyArray"):
    ## import the calibration parameters
    path = (os.getcwd() + calib_path)
    with open(path, "r") as read_file:
        calib_params = json.load(read_file)

        transformation_matrix_icp = np.asarray(calib_params["transformation_matrix"])
        delta1_z = calib_params['delta1_z']
        delta2_z = calib_params['delta2_z']
        theta1_xz = calib_params['theta1_xz']
        theta1_yz = calib_params['theta1_yz']
        theta2_xz = calib_params['theta2_xz']
        theta2_yz = calib_params['theta2_yz']
    
    ## Import the to be calibrated parts and converting them into an numpy array in order to process them
    if type(pcl1) == str and filetype == "Pointcloud":
        path_pcl1 = os.getcwd() + "/data/pointclouds/raw/" + pcl1
        pcl_1 = o3d.io.read_point_cloud(path_pcl1)
        npy_1 = np.asarray(pcl_1.points)
    elif pcl1 == None and filetype == "Pointcloud":
        path_pcl1 = os.getcwd() + "/data/pointclouds/raw/" + str(Layerheight) + "_LLS1"
        pcl_1 = o3d.io.read_point_cloud(path_pcl1)
        npy_1 = np.asarray(pcl_1.points)
    elif type(pcl1) == str and filetype == "NumpyArray":
        path_pcl1 = os.getcwd() + "/data/Numpy_Arrays/raw/" + pcl1
        npy_1 = np.load(path_pcl1)
    elif pcl1 == None and filetype == "NumpyArray":
        path_pcl1 = os.getcwd() + "/data/Numpy_Arrays/raw/" + str(Layerheight) + "_LLS1"
        npy_1 = np.load(path_pcl1)
    else:
        npy_1 = pcl1

    if type(pcl2) == str and filetype == "Pointcloud":
        path_pcl2 = os.getcwd() + "/data/pointclouds/raw/" + pcl2
        pcl_2 = o3d.io.read_point_cloud(path_pcl2)
        npy_2 = np.asarray(pcl_2.points)
    elif pcl2 == None and filetype == "Pointcloud":
        path_pcl2 = os.getcwd() + "/data/pointclouds/raw/" + str(Layerheight) + "_LLS2"
        pcl_2 = o3d.io.read_point_cloud(path_pcl2)
        npy_2 = np.asarray(pcl_2.points)
    elif type(pcl2) == str and filetype == "NumpyArray":
        path_pcl2 = os.getcwd() + "/data/Numpy_Arrays/raw/" + pcl2
        npy_2 = np.load(path_pcl2)
    elif pcl2 == None and filetype == "NumpyArray":
        path_pcl2 = os.getcwd() + "/data/Numpy_Arrays/raw/" + str(Layerheight) + "_LLS2"
        npy_2 = np.load(path_pcl2)
    else:
        npy_2 = pcl2
    
    ## Inverting the Z-values making it easier to process them
    npy_1[:,2] = npy_1[:,2]*-1
    npy_2[:,2] = npy_2[:,2]*-1

    ## Creating PointCloud classes (self made class from PCL.py)
    pcl1 = PointCloud(npy_1)
    pcl2 = PointCloud(npy_2)

    ## Creating an Z-area where points are kept (removing outliers)
    pcl1.clean('z', -1000, -100)
    pcl2.clean('z', -1000, -100)

    ## Calibrating with the calibration parameters
    ## Normalize to the minimum in z direction
    pcl1.shift('z', delta1_z)
    pcl2.shift('z', delta2_z)
    ## Correct the tilt in z direction
    pcl1.rotate('x', 'z', theta1_xz)
    pcl1.rotate('y', 'z', theta1_yz)
    pcl2.rotate('x', 'z', theta2_xz)
    pcl2.rotate('y', 'z', theta2_yz)

    pcl1.clean('z', 0.3, 300)
    pcl2.clean('z', 0.3, 300)
    ## Creating again numpy arrays in order to create PointCloud classes (this time we use the class from open3d)
    npy_1 = pcl1.transform_to_np()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(npy_1)

    npy_2 = pcl2.transform_to_np()
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(npy_2)

    ## visualization of the results
    draw_registration_result(pcd1, pcd2, transformation_matrix_icp)

    ## transforming npy3 into the right coordinate system using the generated transformation
    npy1_transformed = np.asarray(pcd1.transform(transformation_matrix_icp).points)

    ## merging of both pointclouds
    merged_pcl = np.append(npy1_transformed, npy_2, axis=0)
    
    ## saving the merged pcl as .npy
    height = Layerheight
    gp.save_as_npArray(Data=merged_pcl, name=str(height), path = "/data/Numpy_Arrays/merged/")
    
    ## saving the merged pcl as .ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pcl)
    pcl_path = os.getcwd() + "/data/pointclouds/merged/" + str(height) + "_merged.ply"
    o3d.io.write_point_cloud(pcl_path, pcd)

    return merged_pcl


if __name__ == "__main__":
    pass
    # ##pcl_1, pcl_2 = calibration_acquisition()
    # ##gp.save_as_npArray(pcl_1, Layerheight="pcl_1_calib", path = "/data/Calibration_PCLs/")
    # ##gp.save_as_npArray(pcl_2, Layerheight="pcl_2_calib", path = "/data/Calibration_PCLs/")
    # file_path = os.path.join(os.environ['ROOT'], 'files/calibration_parameters.json')
    # get_calibration_parameters(
    #     '63848113504e7d63865e576b',
    #     '6384811d504e7d63865e576c',
    #     '63848128504e7d63865e576e',
    #     '63848130504e7d63865e576f',
    #     '63848137504e7d63865e5771',
    #     '6384813d504e7d63865e5772',
    #     file_path,
    #     False
    #     )
    # product = db.initiate_from_db('63847797504e7d63865e5767')
    # user = db.initiate_from_db(os.environ['USER_ID'])
    # nominal = db.State('Calibrated', 'Printjob is done and the parameters are obtained.', product, 3)
    # nominal.post()
    # transf_json = db.Files('Calibration Parameters', 'Transformation parameters to join the data from both sensors.', nominal, user, 'Parameters', localfilepath=file_path)
    # transf_json.post()
