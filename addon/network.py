import os
import pickle
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as Pool
import logging
log = logging.getLogger("SmoPa3D")

from .gcode.parser import Gcode
from .node import Node, Environment
from .command import Command, Layer, calculate_command_mesh
from .decorators import runtime, track

class Network:
    """Class to create and manage Nodes from the given gcode
    @param gcode_path: path to the gcode file
    @param resolution: the concentration of the pointcloud per mm
    @param node_size: the lenght, in mm, of the pointcloud around the node
    @param filament_thickness: the diameter, in mm, of the filament used in the printing simulation
    """
    def __init__(self, gcode_path:str, resolution:float, node_size:float, node_distance:float=0.2, extrusion_multiplier:float=1, saving_path:str='data/simulation/pickle') -> None:
        log.info('Creating network...')
        self.saving_path = saving_path
        os.makedirs(self.saving_path, exist_ok=True)

        self.gcode = Gcode(path=gcode_path)
        self.env = Environment(resolution=resolution, node_size=node_size)
        self.node_distance = node_distance
        self.extrusion_multiplier = extrusion_multiplier
        self.nodes:list[Node] = []
        self.commands:dict[int, Command] = {}
        self.layers:dict[float, Layer] = {}
        self.coords:list[tuple[float, float, float]] = []

        # Get layer height from gcode
        self.layer_height = None
        for command in self.gcode.gcode[:self.gcode.setup_end]:
            if ';Layer height: ' in command:
                self.layer_height = float(command.split(';Layer height: ')[1])
                break
        if self.layer_height is None:
            log.warning('Layer height not found in gcode. Using 0.2 mm as default')
            self.layer_height = 0.2

        self.create_network()

    @runtime
    def create_network(self) -> None:
        """Create the nodes from the gcode commands"""
        empty_commands = 0
        for (command_id, gcode_command) in enumerate(self.gcode.commands):
            if gcode_command.m == 109:  # Set extruder temperature
                self.temperature = gcode_command.s
            if not gcode_command.is_transformable: continue  # Only considers commands that extrude filament

            # Create command and nodes
            command = Command(self, gcode_command, id=command_id)
            if command.qtd_nodes == 0:
                if empty_commands < 15: log.warning(f'Command {command.id} has no nodes')
                empty_commands += 1
                continue

            self.tree = KDTree(self.coords)
        log.warning(f'{empty_commands} ({empty_commands/len(self.commands)*100:.2f}%) commands have no nodes, and will not be simulated')

    def get_layer(self, z:float) -> Layer:
        """Get the layer of the given z coordinate"""
        if z not in self.layers:
            self.layers[z] = Layer(self, z)
        return self.layers[z]

    @runtime
    def simulate_printer(self, node_limit:int=-1, workers:int=-1, optmize_memory:bool=True) -> None:
        """Run the calculations of each node
        @param node_limit: limit the number of nodes to be calculated. If -1, all nodes will be calculated
        @param workers: number of processes to be used in the calculation. If -1, all available cores will be used
        @param optmize_memory: if True, the nodes will be saved after each iteration, and the previous layers will be deleted from memory"""
        log.info('Simulating printer...')

        current_layer = self.nodes[0].layer
        for (i, node) in enumerate(tqdm(self.nodes[:node_limit])):
            obstacles = self.calculate_obstacles(node, workers=workers)
            node.iterate_volume(obstacles)
            node.save()
            
            if node.layer != current_layer:
                if optmize_memory:
                    threshold_in_use = node.layer.z - self.layer_height * 3  # Delete layers that are not in the search radius of the calculate_obstacles function
                    layers_to_wipe = [layer for layer in self.layers.values() if layer.z < threshold_in_use]
                    for layer in layers_to_wipe:
                        layer.wipe_memory()
                current_layer = node.layer

    @track
    def calculate_obstacles(self, node:Node, workers:int) -> np.ndarray:
        search_radius = self.env.node_size / np.sqrt(3)
        query = self.tree.query_ball_point(node.coord, search_radius, p=2, workers=workers)
        # Restrict teh search in the z axis to range between the node and 2 layers below
        neighbours = [self.nodes[x] for x in query if self.nodes[x].z >= node.z - 2 * self.layer_height and self.nodes[x].z <= node.z]
        return node.obstacles(neighbour_nodes=neighbours)  # TODO: processes = workers

    def calculate_meshes(self, processes:int=None) -> None:
        """Calculate the meshes of each command. Pass processes as 0 not to use multiprocessing"""
        log.info('Calculating meshes...')
        if processes == 0:
            for command in tqdm(self.commands.values()):
                command.vertices_filepath, command.faces_filepath = calculate_command_mesh(command)
        else:
            with Pool(processes) as pool:
                meshes = list(tqdm(pool.map(calculate_command_mesh, (list(self.commands.values()))), total=len(self.commands)))
            
            log.info('Assigning meshes to commands...')
            for command, (vertices_path, faces_path) in zip(self.commands.values(), meshes):
                command.vertices_filepath = vertices_path
                command.faces_filepath = faces_path


    def save(self, filename:str="network") -> None:
        """Save the network in a pickle file"""
        if filename[-4:] != ".pkl": filename += ".pkl"
        try:
            with open(os.path.join(self.saving_path, filename), "wb") as fle:
                pickle.dump(self, fle)
            log.info(f'Network saved in {os.path.join(self.saving_path, filename)}')
        except Exception as e:
            log.warning(f'Could not save network due to {e}')

def load_network(path:str="data/simulation/pickle/network.pkl") -> Network:
    """Load the network from a pickle file"""
    with open(path, "rb") as fle:
        return pickle.load(fle)

if __name__ == '__main__':
    net = Network('test/sample.gcode', 0.01, 2)
    net.simulate_printer()