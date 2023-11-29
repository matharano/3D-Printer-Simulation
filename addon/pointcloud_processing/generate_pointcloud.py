import numpy as np
import open3d as o3d
import pandas as pd
import os

## Generation of a pandas DataFrame and preprocessing of the data
def generate_df(xyz_profile):
    data = np.reshape(xyz_profile, (-1, 4))
    df = pd.DataFrame(data)
    df = df.set_axis(("X", "Y", "Z", "Intensität"), axis='columns')
    return df

### Generates a Pointcloud out of a pandas DataFrame and saves it as .ply
def generate_pcl(Data, path="/data/pointclouds/raw/", name = "unknownName", LLS = "unknownLLS"):
    if type(Data) != np.ndarray:
        df = Data
        xyz= df.iloc[:,[0, 1, 2]].to_numpy()
    else:
        xyz= Data[:,[0, 1, 2]]
    xyz = np.reshape(xyz, (-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    filepath = os.path.join(os.environ['ROOT'], path, str(name) + LLS + ".ply")
    o3d.io.write_point_cloud(filepath, pcd)
    print("Profil wurde als pcl gespeichert!")
    return filepath

### Saves a pandas DataFrame as hdf5
def save_as_hdf(DataFrame, path = "/data/DataFrame/Ergebnisse.h5"):
    df = DataFrame
    filepath = os.path.join(os.environ['ROOT'], path)
    df.to_hdf(filepath, key="DataFrame")
    print("Profil wurde als .HDF5 gespeichert!")

def save_as_npArray(Data, name, LLS, path = "/data/Numpy_Arrays/") -> str:
    """Return the path to the saved file"""
    Array = Data
    filepath = os.path.join(os.environ['ROOT'], path, str(name) + LLS + ".npy")
    np.save(filepath,Array)
    return filepath

if __name__ == "__main__":
    path = "D:/DATA/"

    data = np.array([])
    df2 = pd.DataFrame(columns=["X", "Y", "Z", "Sensor", "Intensität"])

    samples = os.listdir(path)
    i = 0 
    for sample in samples:
        print("Layer " + str(i))
        sample_path = str(path + str(i) + ".npy")
        array = np.load(sample_path)

        df = generate_df(array)
        df = df[df.Intensität != 0]
        df = df[df.Z > 20]
        df_max = df.quantile(q=0.2)
        df = df[df.Z < (df_max["Z"])]
        df2 = pd.concat([df2, df])
        i += 1

    generate_pcl(df2)