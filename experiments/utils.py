import logging
import matplotlib.pyplot as plt
import numpy as np

def plot_pointclouds(pointcloud:np.array, markersize:float=0.01) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, figsize=(10,10))
    fig.suptitle("Raw data view")
    fig.tight_layout(pad=3.0)
    
    # XY
    ax[0].plot(pointcloud[:,1], pointcloud[:,0], 'o', markersize=markersize)
    # ax[0].set_aspect('equal')
    ax[0].title.set_text('YX')
    ax[0].set_xlabel('Y')
    ax[0].set_ylabel('X')

    # XZ
    ax[1].plot(pointcloud[:,0], pointcloud[:,2], 'o', markersize=markersize)
    ax[1].set_aspect('equal')
    ax[1].title.set_text('XZ')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Z')

    # YZ
    ax[2].plot(pointcloud[:,1], pointcloud[:,2], 'o', markersize=markersize)
    ax[2].set_aspect('equal')
    ax[2].title.set_text('XZ')
    ax[2].set_xlabel('Y')
    ax[2].set_ylabel('Z')

    return fig, ax

def crop_to_roi(pointcloud:np.array, roi:list[list[float]]):
    """
    Crop pointcloud to region of interest (roi)
    roi: [[min x, max x], [min y, max y], [min z, max z]]
    """
    pcl = pointcloud.copy()
    if roi[0][0] is not None: pcl = pcl[pcl[:, 0] > roi[0][0]]  # min x
    if roi[0][1] is not None: pcl = pcl[pcl[:, 0] < roi[0][1]]  # max x
    if roi[1][0] is not None: pcl = pcl[pcl[:, 1] > roi[1][0]]  # min y
    if roi[1][1] is not None: pcl = pcl[pcl[:, 1] < roi[1][1]]  # max y
    if roi[2][0] is not None: pcl = pcl[pcl[:, 2] > roi[2][0]]  # min z
    if roi[2][1] is not None: pcl = pcl[pcl[:, 2] < roi[2][1]]  # max z
    return pcl

def init_logging(log_level='WARNING'):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(module)-10s] %(message)s", datefmt='%d/%m/%Y %H:%M:%S')
    rootLogger = logging.getLogger("SmoPa3D")

    fileHandler = logging.FileHandler("smopa3d.log")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.WARNING)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(log_level)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)
    return rootLogger