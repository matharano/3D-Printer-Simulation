import sys
import logging
import numpy as np
running_in_blender = 'Blender' in sys.executable
if running_in_blender:
    import bpy

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

def color_range(slices:int) -> list:
    """Returns a list of colors for the given number of slices"""
    def red(i:int) -> float:
        """graph = \_"""
        # return max(0, 1-2*i/slices)
        return 0

    def green(i:int) -> float:
        """graph = _/"""
        # return max(0, 2*i/slices-1)
        return max(i/slices - 0.5, 0) * 2

    def blue(i:int) -> float:
        """graph = /\\"""
        # return 1-(red(i)+green(i))
        return min(1, 0.2 + 0.8 * 2 * i/slices) - max(i/slices - 0.5, 0) * 2
    
    colors = []
    for slice in range(slices):
        color_values = (red(slice), green(slice), blue(slice), 1)
        color = bpy.data.materials.new(f"Color_{slice}")
        color.use_nodes = True
        tree = color.node_tree
        nodes = tree.nodes
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color_values
        color.diffuse_color = color_values
        colors.append(color)
    return colors

def npy_to_ply(path:str):
    import open3d as o3d
    xyz = np.load(path)
    xyz = np.reshape(xyz, (-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    filepath = path.replace(".npy", ".ply")
    o3d.io.write_point_cloud(filepath, pcd)