import random
import numpy as np
from .utils import running_in_blender
if running_in_blender:
    import bpy
    import bmesh


class PointCloud:
    """Class for handling pointclouds."""
    def __init__(self, path:str) -> None:
        self.pcl:np.ndarray = np.load(path)
        self.correct_z_axis()

    def correct_z_axis(self) -> float:
        """Invert z axis, then get the height of the bed from the pointcloud and move pointcloud so that the bed is at z=0."""
        self.pcl[:, 2] = -self.pcl[:, 2]
        z_values = self.pcl[:, 2].copy()
        z_values.sort()
        mode = z_values[int(len(z_values)*0.4):int(len(z_values)*0.6)]
        bed_z = mode.mean() - 5* mode.std()

        self.pcl[:, 2] = self.pcl[:, 2] - bed_z
        return bed_z

    def crop_ROI(self, x:tuple[float, float]=(None, None), y:tuple[float, float]=(None, None), z:tuple[float, float]=(None, None)) -> np.ndarray:
        """Crop the pointcloud to the region of interest."""
        if x[0] is not None: self.pcl = self.pcl[self.pcl[:, 0] > x[0]]
        if x[1] is not None: self.pcl = self.pcl[self.pcl[:, 0] < x[1]]
        if y[0] is not None: self.pcl = self.pcl[self.pcl[:, 1] > y[0]]
        if y[1] is not None: self.pcl = self.pcl[self.pcl[:, 1] < y[1]]
        if z[0] is not None: self.pcl = self.pcl[self.pcl[:, 2] > z[0]]
        if z[1] is not None: self.pcl = self.pcl[self.pcl[:, 2] < z[1]]
        return self.pcl

    def downsample(self, select_rate:float=0.05) -> np.ndarray:
        # Decrease amount of points
        sample = random.sample(range(len(self.pcl)), int(len(self.pcl) * select_rate))
        self.pcl = self.pcl[sample]
        return self.pcl
    
    def place_pointcloud(self, name:str, parent:bpy.types.Object, move:tuple[float, float, float]=None) -> bpy.types.Object:
        """Plot pointcloud given by x, y and z data in blender as points"""
        mesh_data = bpy.data.meshes.new(name)
        bm = bmesh.new()

        for v in self.pcl[:, :3]:
            bm.verts.new(v)
        
        bm.to_mesh(mesh_data)
        mesh_obj = bpy.data.objects.new(mesh_data.name, mesh_data)

        parent.parent = mesh_obj
        parent.parent_type = 'OBJECT'

        # Move pointcloud
        if move is not None:
            mesh_obj.location = move

        # Visuals
        mesh_obj.display.show_shadows = False
        mesh_obj.show_all_edges = False
        mesh_obj.show_instancer_for_viewport = False
        return mesh_obj