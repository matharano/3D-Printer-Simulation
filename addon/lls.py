from __future__ import annotations
import bpy
from mathutils import Vector
import numpy as np
import logging
import itertools
from tqdm import tqdm
from tqdm.contrib.itertools import product
import concurrent.futures

from . import main

log = logging.getLogger('SmoPa3D')

class LaserLineScannerOperator(bpy.types.Operator):
    bl_idname = "fdm_simulator.scan"
    bl_label = "Simulate Laser Line Scanner"

    def execute(self, context):
        pcl = scan(draw=False, x_resolution=0.05, y_resolution=0.05, y_range=[40, 60])
        np.save("virtual_scan.npy", pcl)
        return {'FINISHED'}

def scan(
    x_range: tuple[float, float] = (95.306, 97.306),
    x_resolution: float = 0.02,
    y_range: tuple[float, float] = (-10, 10),
    y_resolution: float = 1,
    angle_range: tuple[float, float] = (-0.243, 0.243),
    z: float = 130,
    draw: bool = False
    ) -> np.ndarray:
    """Simulate the laser line scanner. Returns the point cloud of the scan in the format `np.array([[x0, y0, z0], ..., [xn, yn, zn]])`.\n
    Arguments:
    @param x_range: The limits of the laser line scanner source in the x axis
    @param x_resolution: The resolution of the x axis. Resolution of the lls is 0.003 ~ 0.05 mm according to the datasheet
    @param y_range: The range of the y axis
    @param y_resolution: The resolution of the y axis. Resolution of the encoder is
    @param angle_range: The range of the angle of the lls. Default value assessed experimentally
    @param z: The height of the lls. Default value assessed experimentally
    @param draw: Whether to draw the beams and the point cloud in the scene
    """
    bpy.context.view_layer.update()

    width = 2 * z * np.tan(max(angle_range)) + abs(x_range[1] - x_range[0])
    x_divisions = round(width / x_resolution)
    x_range = np.linspace(*x_range, x_divisions)
    y_range = np.arange(*y_range, y_resolution)
    angle_range = np.linspace(*angle_range, x_divisions)
    log.info("Casting beams")
    beams = [Beam((x, y, z), (angle, 0, -1)) for (x, angle), y in product(zip(x_range, angle_range), y_range)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda beam: beam.cast(), beams)

    if draw:
        log.info("Drawing beams")
        for beam in tqdm(Beam._instances):
            beam.draw()

    pointcloud = np.zeros((len(Beam._instances), 3))
    i = 0
    for beam in Beam._instances:
        if beam.is_hit and beam.is_read:
            pointcloud[i] = beam.hit_position
            i += 1
    pointcloud = pointcloud[:i]

    del Beam._instances[:]
    return pointcloud

class Beam:
    """Class for handling the beam of the laser line scanner."""
    id_iter = itertools.count()  # Iterator for assigning unique ids to beams
    _instances:list[Beam] = []  # List of all instances of this class

    def __init__(self, origin:tuple[float, float, float], direction:tuple[float, float, float]) -> None:
        self.origin = origin
        self.direction = direction
        self.id = next(self.id_iter)
        self.detector_offset = (0, -65, 0)  # The offset of the detector from the origin of the beam
        self.detector_position = Vector(self.origin) + Vector(self.detector_offset)
        Beam._instances.append(self)
    
    def __getitem__(self, id:int) -> Beam:
        return self._instances[id]
        
    def cast(self) -> tuple[bool, tuple[float, float, float]]:
        """Cast the beam and return the result."""
        reading = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, Vector(self.origin), Vector(self.direction))
        self.is_hit, self.hit_position, self.hit_normal, self.hit_index, self.hit_object, self.hit_matrix = reading
        
        # Check if the beam can be seen by the detector
        if self.is_hit:
            detector_direction =  self.detector_position - Vector(self.hit_position)
            detector_direction.normalize()
            reading = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, self.hit_position + Vector((0, 0, 0.001)), detector_direction)
            self.is_read = not reading[0]  # If the beam is blocked by an object
            if not self.is_read:
                self.detector_position = reading[1]
        return reading
    
    def draw(self) -> None:
        """Draw the path of the beam and the point where it hits."""
        if not self.is_hit: return
        color = green() if self.is_read else red()
        if self.is_hit:
            # Draw an icosphere where the beam hits
            hit_pointer = reference_icosphere()
            hit_pointer.location = self.hit_position
            hit_pointer.active_material = color
            main.group(hit_pointer, f"Laser beams/{self.id}")

            # Draw the beam path
            emission_beam = main.add_bezier(self.origin, self.hit_position)
            emission_curve = emission_beam.data
            emission_curve.dimensions = '3D'
            emission_curve.bevel_depth = 0.02
            emission_curve.bevel_resolution = 3
            emission_curve.materials.append(color)
            emission_curve.name = "Beam path"
            main.group(emission_beam, f"Laser beams/{self.id}")
            reading_beam = main.add_bezier(self.hit_position, self.detector_position)
            reading_curve = reading_beam.data
            reading_curve.dimensions = '3D'
            reading_curve.bevel_depth = 0.02
            reading_curve.bevel_resolution = 3
            reading_curve.materials.append(color)
            reading_curve.name = "Beam path"
            main.group(reading_beam, f"Laser beams/{self.id}")

def reference_icosphere() -> bpy.types.Object:
    """Create a reference icosphere and return it."""
    if "Reference Icosphere" in bpy.data.objects:
        icosphere = bpy.data.objects["Reference Icosphere"]
        icosphere = icosphere.copy()
        return icosphere
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.02, enter_editmode=False, location=(0, 0, 0))
    icosphere = bpy.context.object
    icosphere.data.name = "Reference Icosphere"
    try:
        bpy.context.scene.collection.children["Collection"].objects.unlink(icosphere)
    except:
        pass

    icosphere = icosphere.copy()
    return icosphere

def red():
    material = bpy.data.materials.new("Red")
    material.diffuse_color = (1, 0, 0, 1)
    return material

def green():
    material = bpy.data.materials.new("Green")
    material.diffuse_color = (0, 1, 0, 1)
    return material