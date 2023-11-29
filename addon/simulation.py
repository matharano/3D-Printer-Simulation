import bpy
import bmesh
import os
import numpy as np
from tqdm import tqdm
import logging

from . import geometry as geo
from .node import Node
from .network import Network, load_network
from .pointcloud import PointCloud
from . import utils

log = logging.getLogger('SmoPa3D')

class SimulateOperator(bpy.types.Operator):
    bl_idname = "fdm_simulator.simulate"
    bl_label = "Simulate an FDM 3D printer"

    def execute(self, context):
        net = Network('test/benchy-02infill.gcode', 0.02, 1, node_distance=2/3)
        net.simulate_printer()
        net.calculate_meshes(processes=None)
        net.save("benchy.pkl")

        log.info('Plotting...')
        for command in tqdm(net.commands.values()):
            if command.vertices is None or len(command.faces) == 0: continue
            node_mesh = place_mesh(command.vertices, command.faces, f'Command {command.id}')
            group(node_mesh, f'Simulation/layer {command.layer.z}')  # Group by layer
        return {'FINISHED'}

class DrawSimulationOperator(bpy.types.Operator):
    bl_idname = "fdm_simulator.draw_simulation"
    bl_label = "Draw an already simulated FDM 3D printed part"

    def execute(self, context):
        net = load_network('data/simulation/over/benchy.pkl')

        log.info('Plotting...')
        for command in tqdm(net.commands.values()):
            if command.vertices is None or len(command.faces) == 0: continue
            node_mesh = place_mesh(command.vertices, command.faces, f'Command {command.id}')
            group(node_mesh, f'Simulation/layer {command.layer.z}')  # Group by layer
        return {'FINISHED'}
    
class LoadScanOperator(bpy.types.Operator):
    bl_idname = "fdm_simulator.load_scan_point_cloud"
    bl_label = "Load Scan in point cloud format"
    
    def execute(self, context):
        path = "data/awk_wzl-printed_in_windows-2_lls"
        scans = {int(os.path.split(fle)[-1].split('_')[0]): os.path.join(path, fle) for fle in os.listdir(path) if fle[-5] == '2'}
        layers = sorted(scans.keys())
        colors = utils.color_range(len(scans))
        for layer, color in tqdm(zip(layers[::-1], colors)):
            fle = scans[layer]
            pointcloud = PointCloud(fle) 
            pointcloud.crop_ROI((0, 25), (23, 33), (0, 10))
            pointcloud.downsample(0.5)
            
            # Create an icosphere to use as an instance
            bpy.ops.mesh.primitive_cube_add(size=0.05)
            icosphere = bpy.context.object
            icosphere.active_material = color

            # Place instances at each vertex of the point cloud
            placed_pointcloud = pointcloud.place_pointcloud(f'Layer {layer}', icosphere, (97.615, -10.637, 0))
            group(placed_pointcloud, 'Scan')

        return {'FINISHED'}

class LoadScanMeshOperator(bpy.types.Operator):
    bl_idname = "fdm_simulator.load_scan_mesh"
    bl_label = "Load Scan as a mesh"
    
    def execute(self, context):
        path = "data/awk_wzl-printed_in_windows-2_lls"
        scans = {int(os.path.split(fle)[-1].split('_')[0]): os.path.join(path, fle) for fle in os.listdir(path) if fle[-5] == '2'}
        layers = sorted(scans.keys())
        colors = utils.color_range(len(scans))
        for layer, color in tqdm(zip(layers[::-1], colors)):
            fle = scans[layer]
            pointcloud = PointCloud(fle)
            pointcloud.crop_ROI((0, 25), (23, 33), (0, 10))
            pcl = pointcloud.pcl[:, :3]
            vertices, faces = geo.reconstruct_pointcloud_mesh(pcl, 0)
            placed_pointcloud = place_mesh(vertices, faces, f'Layer {layer}')

            group(placed_pointcloud, 'Scan')

        return {'FINISHED'}

def group(obj:bpy.types.Object, groupName:str):
    """Group object by layer. If groupName is passed as a route, all parent layers will be created."""
    folders = os.path.normpath(groupName).split(os.sep)
    parent_folder = bpy.context.scene.collection
    for folder in folders:
        if not folder in parent_folder.children:
            bpy.ops.collection.create(name=folder)
            parent_folder.children.link(bpy.data.collections[folder])
        parent_folder = parent_folder.children[folder]
    parent_folder.objects.link(obj)

def place_mesh(pointcloud:np.array, faces:np.array, name:str) -> bpy.types.Object:
    """Plot pointcloud given by x, y and z data in blender as mesh"""
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(pointcloud, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bm.to_mesh(mesh_data)
    mesh_obj = bpy.data.objects.new(mesh_data.name, mesh_data)
    return mesh_obj

def place_node(node:Node) -> None:
    if node.placed_filament is None: return
    vertices, faces = geo.calculate_mesh(node.placed_filament, 16)
    if len(vertices) == 0:
        log.warning(f"Node {node} could not be plotted due to lack of normals.")
        return
    vertices *= node.network.env.resolution
    positioned_node = vertices - np.repeat((1+ node.network.env.node_grid_size)//2, 3) * node.network.env.resolution + node.coord
    node_mesh = place_mesh(positioned_node, faces, 'Node')
    group(node_mesh, 'Layer ' + str(round(node.z, 2)))  # Group by layer

def place_node_by_close_points(node:Node) -> None:
    """Calculate mesh by finding points close to each other"""
    if node.placed_filament is None: return
    vertices = geo.calculate_shell(node.placed_filament)
    vertices = geo.downsample(vertices, 0.5)
    faces = geo.find_triangles(vertices, 2)
    positioned_node = vertices - np.repeat((1+ node.network.env.node_grid_size)//2, 3) + node.coord / node.network.env.resolution
    node_mesh = place_mesh(positioned_node, faces, 'Node')
    group(node_mesh, 'Layer ' + str(round(node.z, 2)))  # Group by layer

def place_obstacles(node:Node, net:Network) -> None:
    search_radius = net.env.node_size * np.sqrt(2)
    query = net.tree.query_ball_point(node.coord, search_radius)
    neighbours:list[Node] = [net.nodes[x] for x in query]

    for nbr in neighbours:
        if nbr.placed_filament is None and nbr != node: continue
        log.debug(f'Plotting node: {nbr} ...')
        shift_vector = np.around(nbr.coord/node.network.env.resolution) - np.around(node.coord/node.network.env.resolution)
        shifted_node = geo.shift_array(nbr.placed_filament, shift_vector)
        vertices, faces = geo.calculate_mesh(shifted_node, 16)
        if len(vertices) == 0: continue
        positioned_node = vertices - np.repeat((1+ node.network.env.node_grid_size)//2, 3) + node.coord / node.network.env.resolution
        node_mesh = place_mesh(positioned_node, faces, 'Neighbour')

    log.debug('Plotting nozzle...')
    vertices, faces = geo.calculate_mesh(node.network.env.nozzle, 16)
    positioned_node = vertices - np.repeat((1+ node.network.env.node_grid_size)//2, 3) + node.coord / node.network.env.resolution
    node_mesh = place_mesh(positioned_node, faces, 'Nozzle')
    group(node_mesh, 'Nozzle')
 
    log.debug('Plotting bed...')
    bed = node.network.env.bed(node.z)
    if len(bed != 0) > 0:
        vertices, faces = geo.calculate_mesh(bed, 16)
        positioned_node = vertices - np.repeat((1+ node.network.env.node_grid_size)//2, 3) + node.coord / node.network.env.resolution
        node_mesh = place_mesh(positioned_node, faces, 'Bed')
        group(node_mesh, 'Bed')