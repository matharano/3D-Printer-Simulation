import os
import numpy as np
from skimage.measure import marching_cubes
import logging
log = logging.getLogger("SmoPa3D")

from .gcode.parser import GcodeCommand
from .node import Node, join_nodes, assign_values
from .geometry import width_model

class Command:
    """Represents a command from the gcode file"""
    def __init__(self, network, gcode:GcodeCommand, id:int) -> None:
        self.network = network
        self.gcode = gcode
        self.id = id
        self.nodes:list[Node] = []
        self._vertices:np.ndarray = None
        self._faces:np.ndarray = None
        self.vertices_filepath:str = None
        self.faces_filepath:str = None

        # Update network to include this command
        self.network.commands[id] = self

        # Calculate distribution of nodes along the command
        self.start = np.array((self.gcode.last_position['x'], self.gcode.last_position['y'], self.gcode.last_position['z']))
        self.end = np.array((self.gcode.x if self.gcode.x is not None else self.gcode.last_position['x'], self.gcode.y if self.gcode.y is not None else self.gcode.last_position['y'], self.gcode.z if self.gcode.z is not None else self.gcode.last_position['z']))
        self.trajectory_length = np.linalg.norm(self.end - self.start)

        # Calculate properties
        self.e = self.gcode.e
        self.feedrate = (self.e - self.gcode.last_ocurrence('e')) / self.trajectory_length
        self.speed = self.gcode.f if self.gcode.f is not None else self.gcode.last_ocurrence('f')

        nominal_width = width_model(self.network.temperature, self.feedrate, self.speed)
        self.node_distance = nominal_width * self.network.node_distance
        self.qtd_nodes = round(self.trajectory_length / self.node_distance) + 1
        self.nodes_coords = np.linspace(self.start, self.end, self.qtd_nodes)
        if (self.start[0], self.start[1], self.start[2]) in self.network.coords:  # Remove coincident starts and endings
            self.nodes_coords = self.nodes_coords[1:]
            self.qtd_nodes -= 1

        if self.qtd_nodes <= 0: return
        # volume_distribution = volume_profile(self.qtd_nodes, self.trajectory_length, self.speed) * self.total_extruded_volume
        volume_distribution = np.ones(self.qtd_nodes)/self.qtd_nodes * self.total_extruded_volume

        # Assign layer
        self.layer = self.network.get_layer(self.start[2])

        # Initialize nodes
        for coord, volume in zip(self.nodes_coords, volume_distribution):
            Node(self.network, self, self.layer, coord[0], coord[1], coord[2], volume)

    @property
    def total_extruded_volume(self,
                        constant:float = 0.0025726112346777796,
                        feedrate_multiplier:float = 1.812969202377806
                        ) -> float:
        """Volume, in mm^3, of the filament deposited in the node
        @param constant: constant value to calculate the area of the profile. Value retrieved experimentally
        @param feedrate_multiplier: multiplier value to calculate the area of the profile. Value retrieved experimentally"""
        area = constant + feedrate_multiplier * self.feedrate
        volume = area * self.trajectory_length * self.network.extrusion_multiplier
        # volume = self.feedrate * self.trajectory_length * np.pi * 1.75 ** 2 / 4 * self.network.extrusion_multiplier
        return volume
    
    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the mesh. If the vertices are not saved, returns None"""
        if self._vertices is None:
            if self.vertices_filepath is not None:
                return np.load(self.vertices_filepath)
            else:
                return None
        else:
            return self._vertices
    
    @property
    def faces(self) -> np.ndarray:
        """Vertices of the mesh. If the vertices are not saved, returns None"""
        if self._faces is None:
            if self.faces_filepath is not None:
                return np.load(self.faces_filepath)
            else:
                return None
        else:
            return self._faces

def volume_profile(
    qtd_nodes:int,
    distance:float=100,
    nozzle_speed:float=1000,
    jerk:float=20,
    acceleration:float=500
    ):
    times = []
    umax = min(nozzle_speed, np.sqrt(jerk ** 2 + acceleration * distance))
    s_ramp = (umax ** 2 - jerk ** 2) / (2 * acceleration)
    for s in np.linspace(0, distance, qtd_nodes+1)[1:]:
        if s <= s_ramp:
            roots = np.roots((acceleration/2, jerk, -s))
            roots = min(roots[roots > 0].real)
            times.append(roots)
        elif s <= (distance - s_ramp):
            times.append((umax - jerk) / acceleration + (s - s_ramp) / umax)
        else:
            roots = np.roots((-acceleration/2, umax, (distance - s_ramp - s)))
            roots = min(roots[roots > 0].real)
            times.append((umax - jerk) / acceleration + (distance - 2*s_ramp)/umax + roots)
    total_time = times[-1]
    discretization = [times[i] - times[i-1] if i > 0 else times[i] for i in range(len(times))]
    return (discretization / total_time).astype(float)

def calculate_command_mesh(command:Command) -> tuple[str, str]:
    """Calculate the mesh of the command. Returns the saved path of the vertices and faces"""
    if command.qtd_nodes <= 0:
        return None, None
    elif command.qtd_nodes == 1:
        if command.nodes[0].pointcloud is None: return None, None
        joined_pcl = command.nodes[0].pointcloud
        command.nodes[0].wipe()
    else:  # If there are more than one node, join the pointclouds into one collection of points (shape = (n, 3) [x, y, z])
        joined_pcl = np.vstack([join_nodes(command.nodes[0], node) for node in command.nodes]).astype(np.int64)
        for node in command.nodes:
            node.wipe()

    if joined_pcl is None or len(joined_pcl) == 0:
        return None, None
    
    # Shift the pointcloud to remove the negative values
    shift = joined_pcl.min(0) - [1, 1, 1]
    joined_pcl -= shift
    volume_size = (joined_pcl.max(0) + [5,5,5])[[2,1,0]]

    # Populate the voxel with the pointcloud
    voxel = assign_values(np.zeros(volume_size, dtype=bool), joined_pcl)
    
    # Calculate the mesh
    vertices, faces, _, _ = marching_cubes(voxel)
    
    # Positioning in the environment
    vertices = (vertices.astype(np.float32)[:, ::-1] + shift - command.network.env.node_grid_size // 2  - 1) * command.network.env.resolution + command.nodes[0].coord
    
    # Save the mesh to disk so it does not occupy memory
    vert_path = os.path.join(command.network.saving_path, "commands", f"{command.id}", "vertices.npy")
    face_path = os.path.join(command.network.saving_path, "commands", f"{command.id}", "faces.npy")
    os.makedirs(os.path.dirname(vert_path), exist_ok=True)
    os.makedirs(os.path.dirname(face_path), exist_ok=True)
    np.save(vert_path, vertices)
    np.save(face_path, faces)
    del vertices, faces, joined_pcl
    return vert_path, face_path

class Layer:
    """Represents a layer of the print"""
    def __init__(self, network, z:float) -> None:
        self.network = network
        self.z = z
        self.commands:list[Command] = []
        self.nodes:list[Node] = []

    def wipe_memory(self) -> None:
        """Delete the nodes from the memory"""
        for node in self.nodes:
            node.wipe()
        log.info(f'Layer {self.z} wiped')

