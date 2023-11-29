from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .network import Network
    from .command import Command, Layer

from dataclasses import dataclass
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor as Pool
from functools import partial

from . import geometry as geo

log = logging.getLogger('SmoPa3D')

@dataclass
class Environment:
    resolution: float  # Size, in mm, of the edge of the cubic grid unit
    node_size: float  # Size, in mm, of the edge of the cubic volume around the node
    # Node grid size must be uneven

    def __post_init__(self) -> None:
        # Create the volume
        self.node_grid_size = round(self.node_size / self.resolution)
        if self.node_grid_size % 2 == 0: self.node_grid_size += 1  # Node grid size must be uneven
        self.volume = np.zeros((self.node_grid_size, self.node_grid_size, self.node_grid_size))
        self.calculate_nozzle()

    def calculate_nozzle(self, D:float=0.4, angle:float=0.716, height:float=2.5) -> np.array:
        """Create the 3D model of the nozzle in the environment grid.
        ----
        Arguments
        ----
        @param D: Diameter of nozzle, in mm
        @param angle: angle of the trunk of cone of the nozzle, in rad"""
        z0 = self.node_grid_size // 2  # Tip of the nozzle is always the center of the volume
        height = height / self.resolution
        radius = D / (2 * self.resolution) 
        B = 1 / np.tan(angle)
        A = radius - z0 * B
        final_diameter = A + (height + z0) * B

        def nozzle_2d(z:int) -> float:
            """Build the nozzle in 2D.
            @param z: coordinate in grid units"""
            if z < z0:
                return 0
            elif z < z0 + height:
                return A + z * B
            else:
                return final_diameter
        
        revolutionized = geo.apply_revolution(nozzle_2d, self.node_grid_size, self.node_grid_size).astype(int)
        self.nozzle = np.where(revolutionized > 0)
        self.nozzle = np.array(self.nozzle[::-1]).transpose()
        return self.nozzle

    def bed(self, node_z:float) -> np.array:
        """Create the 3D model of the printing bed in the environment grid placing the bed according to the
        position in z of the node.
        ----
        Arguments
        ----
        @param node_z: position, in mm, between the bottom of the nozzle  in the node (that corresponds to the
        top of the printed filament)"""
        z0 = self.node_grid_size//2 - 1 - round(node_z/self.resolution)
        bed = np.zeros_like(self.volume)
        if z0 > 0:
            bed[:z0] = 1
        return bed

class Node:
    def __init__(self, network:Network, command:Command, layer:Layer, x:float, y:float, z:float, filament_volume:float) -> None:
        """@param index: index of the node in the network
        @param x: Coordinate position of the center of the node
        @param y: Coordinate position of the center of the node
        @param z: Coordinate position of the center of the node
        @param filament_volume: Volume, in mm^3, of the filament deposited in the node"""
        self.network = network
        self.command = command
        self.x = x
        self.y = y
        self.z = z
        self.filament_volume = filament_volume
        self._placed_filament = None  # Voxel reprensenting the volume occupied by the node. It is None until iterate_volume() is computed
        self._pointcloud = None  # Pointcloud of the filament, in (x, y, z) coordinates. It is None until iterate_volume() is computed
        self.active:bool = True  # Set as False when the node is saved and wiped from memory
        self.simulated:bool = False  # Set as True when the node is simulated

        # Update Network to include this node
        self.network.nodes.append(self)
        self.index = len(self.network.nodes) - 1
        self.network.coords.append((self.x, self.y, self.z))
        
        # Update Command to include this node
        self.command.nodes.append(self)

        # Update Layer to include this node
        self.layer = layer
        self.layer.nodes.append(self)

    @property
    def coord(self) -> np.ndarray:
        """(x, y, z)"""
        return np.array((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def save(self) -> None:
        """Save the node data in npy files"""
        if self.active:
            self.saving_path = os.path.join(self.network.saving_path, 'nodes', f'{self.index}')
            os.makedirs(self.saving_path, exist_ok=True)
            np.save(os.path.join(self.saving_path, 'placed_filament.npy'), self.placed_filament)
            np.save(os.path.join(self.saving_path, 'pointcloud.npy'), self.pointcloud)
        else:
            log.warn('Node is already inactive. Cannot save it.')
    
    def load(self) -> None:
        """Load the node from the npy files"""
        if self.saving_path is None:
            log.warning('Node was not saved before or saving path could not be found. Cannot load it.')
            return
        
        self._placed_filament = np.load(os.path.join(self.saving_path, 'placed_filament.npy'), allow_pickle=True)
        self._pointcloud = np.load(os.path.join(self.saving_path, 'pointcloud.npy'), allow_pickle=True)
    
    def wipe(self) -> None:
        """Delete the node from the memory, but keeps it in the network to be loaded back again if necessary"""
        if not self.active: return
        del self._placed_filament
        del self._pointcloud
        self.active = False

    @property
    def placed_filament(self) -> np.ndarray:
        """Voxel reprensenting the volume occupied by the node. If node is inactive, load it from the npy files"""
        if not self.active:
            self.load()
        return self._placed_filament
    
    @property
    def pointcloud(self) -> np.ndarray:
        """Pointcloud of the filament, in (x, y, z) coordinates. If node is inactive, load it from the npy files"""
        if not self.active:
            self.load()
        return self._pointcloud
    
    def backpropagate_feedrate(self, volume:float=None, constant:float=0.0025726112346777796, feedrate_multiplier:float=1.812969202377806) -> float:
        """Calculate the feedrate that would be necessary to deposit the given volume of filament
        @param volume: filament volume, in mm^3
        @param constant: constant value to calculate the area of the profile. Value retrieved experimentally
        @param feedrate_multiplier: multiplier value to calculate the area of the profile. Value retrieved experimentally"""
        area = volume / self.command.trajectory_length * self.command.qtd_nodes / self.network.extrusion_multiplier
        return (area - constant) / feedrate_multiplier
        # return volume / (self.command.trajectory_length / self.command.qtd_nodes * np.pi * 1.75 ** 2 / 4 * self.network.extrusion_multiplier)
    
    def draw_revolutionized_profile(self, ground:int, volume:float=None) -> np.ndarray:
        """Draw a drop of the profile with the given volume.
        @param ground: z coordinate of the ground level
        @param volume: filament volume, in mm^3"""
        if volume is None:
            volume = self.filament_volume
        volume_multiplier = volume / self.filament_volume

        self.applied_feedrate = self.backpropagate_feedrate(volume)
        self.width = geo.width_model(self.network.temperature, self.applied_feedrate, self.command.speed)
        self.length = 1.5 * np.cbrt(volume_multiplier) * self.command.trajectory_length / self.command.qtd_nodes
        self.height = 6 *  volume / (np.pi * self.width * self.length)

        h = round(self.height / self.network.env.resolution / 2)
        w = round(self.width / self.network.env.resolution / 2)
        l = round(self.length / self.network.env.resolution / 2)
        grid = self.network.env.volume.copy()
        if any(np.array(grid.shape) < np.array((w, l, h))):
            log.warning(f'Node size is too small for node {self.index}. It was cut to fit in the simulation.')
        center = np.array(grid.shape)//2 - 1
        # if self.height > self.network.layer_height:
        #     center[0] = ground + (center[0] - ground)//2
        # else:
        # center[0] = ground + h
        center[0] = center[0] - round(self.network.layer_height / self.network.env.resolution / 2)
        geo.place_ellipsoid(grid, w, l, h, self.command.end - self.command.start, center=center)
        return grid
    
    def iterate_volume(self, obstacles:np.ndarray, increment_offset:float=0.1, precision:float=0.1, max_iterations:int=30) -> np.ndarray:
        """Calculates the volume that results from the intersection between the deposited filament of the node
        and the environment given by obstacles. Then expands the volume by the interference plus an increment_offset
        and calculates the volume interference again, until the interference volume reaches a value as low as the
        precision, or it gets to the max iterations.
        @param obstacles: an array representing the interacting environment, given by the obstacles method
        @param increment_offset: the extra volume, in percentage, that is increased during each iteration
        to get faster to the result
        @param precision: the acceptable error, in percentage, between the nominal_volume and the actual volume got from the iterations
        @param max_iterations: the maximum size of the loop that is conducted to get to the final volume"""
        
        precision = self.filament_volume * precision  # Convert the units to mm^3

        virtual_volume = self.filament_volume  # In the end of the iterations: virtual volume = self.filament_volume + intersection_volume
        # ground_level = np.where(obstacles[:obstacles.shape[0]//2, obstacles.shape[1]//2, obstacles.shape[2]//2])[0].max()  # Get the ground level of the obstacles
        for it in range(max_iterations):
            if virtual_volume < 0:
                log.warning(f'Negative filament volume in node {self.index}. It is ignored in the simulation.')
                virtual_volume = 0
                return None
            drawn_profile = self.draw_revolutionized_profile(ground=0, volume=virtual_volume)
            molded_profile = 1*((drawn_profile - obstacles) > 0)
            molded_volume = molded_profile.sum() * self.network.env.resolution**3
            diff = self.filament_volume - molded_volume

            if abs(diff) <= precision:
                break
            else:
                virtual_volume += diff * (1 + increment_offset)
                if it == max_iterations - 1:
                    log.warning(f'Node {self.index} did not converge. Difference of {diff*10**6:.0f} μm^3 ({(diff/self.filament_volume)*100:.2f}% difference) between nominal and actual volume.')
        self._placed_filament = molded_profile.astype(int)
        self._pointcloud = np.where(self._placed_filament > 0)
        self._pointcloud = np.array(self._pointcloud[::-1]).transpose()
        self.simulated = True
        return self._placed_filament

    def obstacles(self, neighbour_nodes:list[Node], processes:int=None) -> np.ndarray:
        """Places all obstacles in the grid, including the nozzle, the bed (if applicable) and the adjacent nodes that already have
        their volumes calculated.
        @param neighbour_nodes: list of Node objects that must be considered in the surroundings of the node"""
        if processes is not None and processes == 0:
            neighbours_pts = np.empty((0, 3))
            for nbr in neighbour_nodes:
                neighbours_pts = np.concatenate([neighbours_pts, place_obstacle(self, nbr)])
        else:
            with Pool(max_workers=processes) as pool:
                self_place_obstacle = partial(place_obstacle, self)
                neighbours_pts = np.concatenate(list(pool.map(self_place_obstacle, neighbour_nodes)))
        filter_in_boundaries = np.all((neighbours_pts >= 0) & (neighbours_pts < self.network.env.node_grid_size), axis=1)
        neighbours = assign_values(self.network.env.volume.copy(), neighbours_pts[filter_in_boundaries])
        return (self.network.env.bed(self.z) + neighbours) > 0

def place_obstacle(main_node:Node, obstacle_node:Node) -> np.ndarray:
    neighbours_pts = np.empty((0, 3))
    shift_vector = np.around((obstacle_node.coord - main_node.coord)/main_node.network.env.resolution)
    if any(np.abs(shift_vector) > main_node.network.env.node_grid_size): return np.empty((0, 3))  # Ignore nodes that are too far away (more than the node grid size)
    if obstacle_node.z == main_node.z and (obstacle_node in main_node.command.nodes or obstacle_node.placed_filament is None):  # Add the nozzle of the next and last nodes even if they have not been placed yet
        new_neighbour = obstacle_node.network.env.nozzle + shift_vector
        neighbours_pts = np.concatenate([neighbours_pts, new_neighbour])
    if not obstacle_node.placed_filament is None:  # if the filament has been placed, add it to the neighbours_pts
        new_neighbour = obstacle_node.pointcloud + shift_vector
        neighbours_pts = np.concatenate([neighbours_pts, new_neighbour])
    return neighbours_pts

def assign_values(volume, neighbours_pts) -> np.ndarray:
    """Assigns 1 to the neighbours_pts in the volume array.\n
    Works exactly as `volume[neighbours_pts] = 1` but more efficient."""
    ravel = np.ravel_multi_index(neighbours_pts[:, [2, 1, 0]].astype(int).T, volume.shape)
    np.put(volume, ravel, 1)
    return volume

def join_nodes(main_node:Node, obstacle_node:Node):
    """Join the pointclouds of two nodes, placing the obstacle_node in the position relative to the main_node"""
    if obstacle_node.placed_filament is None: return np.empty((0, 3))
    shift_vector = np.around((obstacle_node.coord - main_node.coord)/main_node.network.env.resolution)
    new_neighbour = obstacle_node.pointcloud + shift_vector
    return new_neighbour