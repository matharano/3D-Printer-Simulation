import os
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.contrib.itertools import product
sns.set()

from .pointcloud_processing.process_pointclouds import layerize_pointclouds
from .pointcloud_processing.calibration import prepare_dataset, execute_global_registration, draw_registration_result

def get_samples(translation:np.ndarray=np.array((0, 0, 0)), rotation:float=30, plot: bool = True):
    square = np.zeros((10, 10))
    square[3:7, 3:7] = 1
    straight = np.array(np.where(square == 1))
    straight = np.vstack((straight, np.zeros((1, len(straight[0])))))

    rotation_matrix = R.as_matrix(R.from_euler('z', rotation, degrees=True))
    transformed = (rotation_matrix @ straight)
    transformed = transformed + translation.reshape(3, 1)

    if plot:
        plot_squares(straight, transformed)

    return straight.T, transformed.T

def get_one_missplaced(deviation:np.ndarray=np.array([-1, 0, 0]), deviation_index:tuple[float, float, float]=[0, 0, 0], plot:bool=True):
    straight, transformed = get_samples(rotation=0, plot=False)
    transformed[deviation_index] += deviation
    if plot:
        plot_squares(straight, transformed)
    return straight, transformed

def plot_squares(pcd1:np.ndarray, pcd2:np.ndarray):
    plt.plot(pcd1[:, 0], pcd1[:, 1], 'o')
    plt.plot(pcd2[:, 0], pcd2[:, 1], 'o')
    plt.legend(['straight', 'transformed'])


def plot3d(pcls:list[np.ndarray], alpha:float=0.4, size_multiplier:float=1, figsize:tuple[int]=(10, 10)) -> plt.Figure:
    """Plots a list of point clouds in 3D"""
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    if not isinstance(pcls, list):
        pcls = [pcls]
    try:
        s = size_multiplier * 0.06/(pcls[0][:, 0].max()-pcls[0][:, 0].min())
    except:
        s = size_multiplier * 0.06
    for pcl in pcls:
        if len(pcl) == 0:
            continue
        x, y, z = pcl[:, 0], pcl[:, 1], pcl[:, 2]
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.scatter(x, y, z, s=s, marker='o', alpha=alpha)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    return fig

def RMSE(reference:np.ndarray, target:np.ndarray, workers:int=-1) -> float:
    octree = KDTree(reference)
    distances, indices = octree.query(target, k=1, p=2, workers=workers)
    rmse = np.sqrt(np.mean(distances**2))
    return rmse

def calculate_accuracy(reference:np.ndarray, target:np.ndarray, threshold:float, workers:int=-1) -> tuple[float, np.ndarray]:
    """Calculates the accuracy of the target point cloud compared to the reference point cloud.
    If reference and target are swept, metric is called 'completeness'.
    Returns the accuracy and the indices of the inaccurate points, i.e. the points of target that do not match any of the reference."""
    octree = KDTree(reference)
    distances, indices = octree.query(target, k=1, distance_upper_bound=threshold, p=2, workers=workers)
    accuracy = np.array(distances <= threshold, dtype=int).sum() / len(distances)
    return accuracy, np.array(range(len(target)))[distances > threshold]

def f1_score(reference:np.ndarray, target:np.ndarray, threshold:float, workers:int=-1, plot_results:bool=False) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculates the F1 score between two pointclouds. Reference and target are interchangeable.
    Returns the score, the indices of the inaccurate points of the reference and the indices of the uncomplete points of the target."""
    accuracy, inaccurate_ids = calculate_accuracy(reference, target, threshold, workers=workers)
    completeness, uncomplete_ids = calculate_accuracy(target, reference, threshold, workers=workers)
    score = 2 * (accuracy * completeness) / (accuracy + completeness)
    if plot_results:
        inaccurate = target[inaccurate_ids]
        accurate = target[np.setdiff1d(np.arange(len(target)), inaccurate_ids)]
        uncomplete = reference[uncomplete_ids]
        complete = reference[np.setdiff1d(np.arange(len(reference)), uncomplete_ids)]
        matched = np.concatenate([accurate, complete], axis=0)
        plot3d([matched, inaccurate, uncomplete])
    return score, inaccurate_ids, uncomplete_ids

def EMD(reference:np.ndarray, target:np.ndarray, workers:int=-1) -> float:
    octree = KDTree(target)
    distances, indices = octree.query(reference, k=1, p=2, workers=workers)
    return distances.sum()

def average_squared_distance(reference:np.ndarray, target:np.ndarray, neighbours:int=1, workers:int=-1) -> np.ndarray:
    """Calculates the average squared distance to neighbours of all target's points to the reference"""
    octree = KDTree(reference)
    distances, indices = octree.query(target, k=neighbours, p=2, workers=workers)
    return distances**2 / neighbours

def k_chamf(reference:np.ndarray, target:np.ndarray, neighbours:int=1, workers:float=-1) -> float:
    """Calculates the k-Nearest Chamfer distance between two point clouds. Reference and target are interchangeable."""
    first_term = average_squared_distance(reference, target, neighbours, workers=workers)
    first_term = np.mean(first_term)
    second_term = average_squared_distance(target, reference, neighbours, workers=workers)
    second_term = np.mean(second_term)
    return first_term + second_term

def load_pointcloud(path:str, bed_path:str, roi:list[list[float]]=[[None, None], [None, None], [None, None]], intensity_threshold:int=60, plot:bool=True) -> np.ndarray:
    """Load the pointcloud from npy file, and layerize it by comparing with its bed.
    @param path: the path to the npy file of the measurement point cloud
    @param bed: the path to the npy file of the bed point cloud
    @param roi: region that will be used. The rest of the part is cut off. Leave it as `None` not to define a limit.
    Format: [[x0, x1], [y0, y1], [z0, z1]]
    @param intensity_threshold: the threshold of the intensity of the point cloud. Points with intensity below this threshold will be removed.
    @param plot: plot the point cloud using matplotlib if requested"""
    pcl =  np.load(path)
    bed = np.load(bed_path)[:, :3]

    pcl = pcl[pcl[:, 3] > intensity_threshold]
    pcl = pcl[:, :3]
    _, pcl = layerize_pointclouds([bed, pcl])  # Layerize pointclouds
    pcl[:, 2] = -pcl[:, 2] + max(pcl[:, 2])  # Invert z axis
    pcl = pcl[pcl[:, 2] < 100]  # Removing outliers
    for i in range(3):
        if roi[i][0]is not None: pcl = pcl[:, i] > roi[i][0]
        if roi[i][1]is not None: pcl = pcl[:, i] < roi[i][1]

    if plot:
        plot3d(pcl)
    
    return pcl

def icp(source:np.ndarray, target:np.ndarray, voxel_size:float=0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs Iteractive Closest Point registration on two point clouds.
    The source point cloud is transformed to match the target point cloud.
    Returns the transformed source point cloud, the target point cloud and the transformation matrix.
    """
    threshold = voxel_size * 0.4

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.reshape(source, (-1, 3)))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.reshape(target, (-1, 3)))

    source_pcl, target_pcl, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source=pcd1, target=pcd2)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcl, target_pcl, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    
    transformation = result_icp.transformation
    return apply_transformation(source, transformation), target, transformation

def apply_transformation(source:np.ndarray, transformation:np.ndarray) -> np.ndarray:
    """Applies a transformation matrix to a point cloud"""
    source = np.hstack((source, np.ones((len(source), 1))))
    source = transformation @ source.T
    source = source.T
    return source[:, :3]

def remove_noise(pointcloud:np.ndarray, threshold:float=0.5, workers:int=-1) -> np.ndarray:
    """Removes noise from a point cloud. Returns the pointcloud without noise and log the deletion ratio, i.e. the percentage of points that were removed."""
    basis = KDTree(pointcloud)
    distances, indices = basis.query(pointcloud, k=[2], p=2, distance_upper_bound=threshold, workers=workers)
    not_noise = distances[:, 0] < threshold
    remove_ratio = 1-not_noise.sum()/len(pointcloud)
    print(f"Removed {remove_ratio*100:.2f}% of the pointcloud")
    return pointcloud[not_noise]

def load_dataset(path="../data/metrics/measurements") -> list[tuple[str, str]]:
    files = []
    for filename in os.listdir(path):
        if not filename.endswith(".npy") or filename.startswith("bed") or int(filename.split(".")[0][-1]) > 1:
            continue
        run = filename.split("_")[1]
        bed = os.path.join(path, f"bed_before_printing_{run}_0.npy")
        files.append((bed, os.path.join(path, filename)))
    return files

def evaluate_metrics(dataset:list[tuple[str, str]], results_path:str="../data/metrics/results") -> None:
    """Evaluates the metrics on a dataset of point clouds."""
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "results.csv"), "w") as f:
        f.write("target,source,rmse,f1,emd,chamfer\n")
    for s, t in product(dataset, dataset):
        s_file = os.path.split(s[1])[-1]
        t_file = os.path.split(t[1])[-1]
        if s == t:
            continue
        elif 'printing_0' in s_file or 'printing_0' in t_file:  # Ignore the first measurement
            continue
        elif 'measurement' not in s_file and 'measurement' not in t_file:  # Only compare scenarios to as-is
            continue

        source = load_pointcloud(s[1], s[0], plot=False)
        target = load_pointcloud(t[1], t[0], plot=False)

        source = remove_noise(source, 0.2)
        target = remove_noise(target, 0.2)

        source, _, _ = icp(source, target)
        
        rmse = RMSE(source, target)
        f1, _, _ = f1_score(source, target, 0.15, plot_results=False)
        emd = EMD(source, target)
        chamfer = k_chamf(source, target)

        with open(os.path.join(results_path, "results.csv"), "a") as f:
            f.write(",".join((
                os.path.split(t[1])[-1],
                os.path.split(s[1])[-1],
                str(round(rmse, 4)),
                str(round(f1, 4)),
                str(round(emd, 4)),
                str(round(chamfer, 4))
                )))
            f.write("\n")

        del source, target