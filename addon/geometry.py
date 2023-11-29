import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import logging

log = logging.getLogger("SmoPa3D")

def apply_revolution(f_of_z, height:int, width:int) -> np.ndarray:
    """Create a solid by revolutioning a function of z over the z-axis.
    @param f_of_z: function that returns a float by given z value (in grid units)
    @param height: length in z of the volume in grid units
    @param width: length in x and y of the volume in grid units"""
    if width % 2 == 0: width += 1  # Volume must be uneven
    max_x = width // 2 + 1
    solid = np.zeros((height, max_x, width), dtype=int)  # z, y, x
    for z in range(height):
        for y in range(max_x):
            if f_of_z(z) < y:
                break
            x_surface = round(np.sqrt(f_of_z(z) ** 2 - y ** 2))
            x_array = [1 if x < x_surface else 0 for x in range(max_x)]
            solid[z, y] = np.array([*x_array[::-1], *x_array[1:]])
    solid = np.hstack((solid[:, ::-1], solid[:, 1:]))
    return solid

def place_ellipsoid(space:np.ndarray, w:int, l:int, h:int, direction:np.ndarray, center:np.ndarray) -> np.ndarray:
    """Function to place an ellipsoid in a 3D space
    @param space: 3D numpy array (environment)
    @param w: width of the ellipsoid
    @param l: length of the ellipsoid
    @param h: height of the ellipsoid
    @param direction: direction of the ellipsoid
    @param center: center of the ellipsoid (z, y, x)"""

    # # Create a rotation matrix from the direction vector
    # enclosed_space = np.zeros((h, max(w, l), max(w, l)))
    h += 1
    direction = direction / np.linalg.norm(direction)
    r  = R.from_euler('z', np.arctan2(direction[1], direction[0]), degrees=False)
    r = R.from_matrix(r.as_matrix()[::-1,::-1].T)

    # Create a grid of points
    z,y,x = np.meshgrid(np.arange(space.shape[0]), np.arange(space.shape[1]), np.arange(space.shape[2]), indexing='ij')

    # Get the points relative to the center of the space
    points = np.vstack((z.ravel(), y.ravel(), x.ravel())) - center.reshape(-1, 1)

    # Rotate the points by the rotation matrix
    points = r.apply(points.T).T

    # Ellipsoid equation
    inside = (points[0, :] / h) ** 2 + (points[1, :] / w) ** 2 + (points[2, :] / l) ** 2 < 0.98
    
    # Cilinder equation
    # inside = ((points[0, :] / h) ** 2 + (points[1, :] / w) ** 2 <= 1) * (abs(points[2, :] / l) <= 1)

    # Set the value of the points inside the ellipsoid to 1
    space.ravel()[inside] = 1
    return space

def calculate_shell(voxel:np.ndarray) -> np.ndarray:
    """Remove the inner points of the geometry"""
    dz, dy, dx = np.gradient(voxel)
    grads = np.absolute(dz) + np.absolute(dy) + np.absolute(dx)
    shell = grads * voxel
    shell = np.where(shell != 0)
    hull = np.array(shell[::-1]).transpose()
    return hull

def calculate_mesh(pointcloud:np.ndarray, simplify_factor:int=32, resolution:float=0.02) -> tuple[np.ndarray]:
    """Calculate and simplify mesh by a given voxel. Returns vertices coordinates and a list of faces, each one given by a list of indexes of the vertices that make part of the face."""
    basis = KDTree(pointcloud)
    radius = resolution * (np.sqrt(3)+0.1)

    neighbours_list = basis.query_ball_point(pointcloud, r=radius, p=2, return_length=False)
    del basis

    gradients = np.zeros((len(pointcloud), 3))
    for i, neighbours in enumerate(neighbours_list):
        if len(neighbours) >=27: continue
        mean = np.mean(pointcloud[neighbours], axis=0)
        gradients[i] = mean - pointcloud[i]
        norm = np.linalg.norm(gradients[i])
        gradients[i] = gradients[i] / norm if norm > 0 else np.zeros(3)

    xyz = np.reshape(pointcloud[gradients.any(1)], (-1, 3))
    normals = np.reshape(gradients[gradients.any(1)] * -1, (-1, 3))
    del gradients, pointcloud
    if len(normals) == 0: return np.zeros((0, 3)), np.zeros((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, scale=2, n_threads=1)
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    # Simplify mesh
    if simplify_factor > 0:
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / simplify_factor
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    pcd.clear()
    mesh.clear()
    densities.clear()
    return vertices, faces

def reconstruct_pointcloud_mesh(pointcloud:np.ndarray, simplify_factor:int=32) -> tuple[np.ndarray]:
    xyz = np.reshape(pointcloud, (-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    last_normals = np.asarray(pcd.normals)
    pcd.normals = o3d.utility.Vector3dVector(np.absolute(last_normals))
    radii = [0.16, 0.32]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

    # Simplify mesh
    if simplify_factor > 0:
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / simplify_factor
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    pcd.clear()
    mesh.clear()
    return vertices, faces

def shift_array(array:np.ndarray, shift_vector:np.ndarray):
    if any([shift > array.shape[i] for (i, shift) in enumerate(shift_vector)]): return np.zeros_like(array)
    H, W, D = array.shape
    dx, dy, dz = shift_vector
    dx, dy, dz = int(dx), int(dy), int(dz)
    augmented_volume = np.zeros((H+abs(dz), W+abs(dy), D+abs(dx)))
    augmented_volume[max(0, dz):H+max(0, dz), max(0, dy):W+max(0, dy), max(0, dx):D+max(0, dx)] = array
    shifted_array = augmented_volume[max(0, -dz):H+max(0, -dz), max(0, -dy):W+max(0, -dy), max(0, -dx):D+max(0, -dx)]
    return shifted_array

def visualize(voxel:np.array) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, y, z = calculate_shell(voxel).transpose()
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def width_model(temperature:float, feedrate:float, speed:float) -> float:
    """Returns the width of the filament in mm for a given temperature, speed and feedrate. Formulation is based on experimental values."""
    root = np.cbrt(feedrate)
    return (
            -1.2411217231463934
            + 0.004062025031923957 * temperature
            - 0.0001331731552701792 * speed
            + 2.984109335460154 * root
            )

def height_model(area:float, width:float) -> float:
    """Returns the height of the filament in mm for a given area and width.
    Calculation based on the area of the ellipse."""
    return 2 * area / (np.pi * width / 2)