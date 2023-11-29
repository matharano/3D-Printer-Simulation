"""
@author: grh, grh_mh

Project: SmoPa3D

Description: This code is used to generate a nominal out of the gcode given by a slicer.
"""

from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import logging
from itertools import repeat
import multiprocessing

log = logging.getLogger('SmoPa3D')

def gcode2dataframe(gcode_file_path:str):
    """Reads a gcode file and transforms it into a pandas dataframe, each row representing a command.\n
    Dataframe format:\n
    ``` bash
    +-------+---------+---------+---------+-----------+---------+-------+
    | index | X-Value | Y-Value | Z-Value | Extrusion | Command | Layer |
    +-------+---------+---------+---------+-----------+---------+-------+
    |  int  |  float  |  float  |  float  |   float   |   int   |  int  |
    """
    with open(gcode_file_path, "r") as gcode:
        data = gcode.read() ### read in the G-Code
        numberOfLayers_start = re.search(";LAYER_COUNT:", data).end()
        numberOfLayers_end = re.search(";LAYER:0", data).start()
        numberOfLayers = int(data[numberOfLayers_start:numberOfLayers_end])

        lines = [line.strip() for line in data.split('\n')]
        data_size = len(lines)
        df = pd.Series(lines)

        gcode_values = np.zeros(((data_size),6))
        amount_extracted_values = 0
        amount_extracted_values_start_layer = 0


        for i in range(numberOfLayers):
            ### Selection of the i-th layer and separation of the code from the entire code block

            startLayer = (";LAYER:" + str(i))
            if i < (numberOfLayers-1):
                startLayerplusone = (";LAYER:" + str(i+1))        
                layer_start = df[df==str(startLayer)].index.values[0]
                layer_end = df[df==str(startLayerplusone)].index.values[0]
                layer = df[(layer_start+1):layer_end]

            elif i == (numberOfLayers-1):
                layer = df[(layer_start+1):]

            ### Separation of only G-code where extrude and process
            layer = layer.drop(layer.loc[layer.str.contains(";", case= False)].index.values)
            layer = layer.drop(layer.loc[layer.str.contains("M20", case= False)].index.values)
            layer = layer.drop(layer.loc[layer.str.contains("Z", case= False)].index.values) ## Skipping commands in which Z changes
            layer = layer.drop(layer.loc[~layer.str.contains("Y", case= False)].index.values)
            layer = layer.drop(layer.loc[~layer.str.contains("X", case= False)].index.values)

            for (j, position) in enumerate(layer):
                ## Extraction of position in X
                X_start = re.search("X", position).end()
                X_end = re.search("Y", position).start()
                X_position = float(position[X_start:X_end])
                ## Extraction of position in Y
                Y_start = re.search("Y", position).end()
                if re.search("E", position) is not None: 
                    Y_end = re.search("E", position).start()
                    Y_position = float(position[Y_start:Y_end])
                else:
                    Y_position = float(position[Y_start:])

                ## Extraction of position in Z
                Z_position = 0.2 + (i*0.2)

                ## Extraction of feedrate
                if (re.search("E", position) is None) != True: 
                    Extrusion_start = re.search("E", position).end()
                    Extrusion = float(position[Extrusion_start:])
                else: Extrusion= float(0.00)
                
                ## Assignment of layer count to the values
                numberOfLayer = i
                
                ## Assignment of the command number to the values
                numberOfCommand = j

                extracted_value = np.array((X_position,Y_position,Z_position, Extrusion, numberOfCommand, numberOfLayer))
                gcode_values[amount_extracted_values: (amount_extracted_values + 1)] = extracted_value
                amount_extracted_values += 1

    gcode_values = gcode_values[:amount_extracted_values]

    df2 = pd.DataFrame(gcode_values)
    df2 = df2.set_axis(("X-Value", "Y-Value", "Z-Value", "Extrusion", "Command", "Layer"), axis='columns')
    return df2

def calculate_profile_dimensions(method:str, D:float=0.4, V_over_U:float=1.58, Rn:float=1, gap:float=0.2, alpha:float=1.505) -> tuple[float, float]:
    """Calculates height and width of the strand. Available options:\n
    * hebda
    * xu
    * comminal

    References:\n
    * [COMMINAL et al., 2018](https://doi.org/10.1016/j.addma.2017.12.013)\n
    * [Hebda et al., 2019](http://hdl.handle.net/10454/16895)\n
    * [Xu et al., 2022](https://hal-mines-paristech.archives-ouvertes.fr/hal-03766358)"""

    if method == 'hebda':
        W = D * alpha * np.sqrt(1/V_over_U)
        H = D / alpha * np.sqrt(1 / V_over_U)
    elif method == 'xu':
        W = D * (1 - Rn/D + np.sqrt((Rn/D - 1)**2 + np.pi * D / (gap * V_over_U) * (Rn/D - 0.5)))
        H = D**2 / (V_over_U * W)
    elif method == 'comminal':
        W = np.pi / 4 * 1 / V_over_U * D**2 / gap + gap * (1 - np.pi / 4)
        H = gap
    else:
        log.warning('Calculation method for profile dimensions do not exist. Possible options are: hebda, xu and comminal')
        W = 0
        H = 0
    return W, H

def draw_profile(geometry:str='ellipse', width:float=0.374109, height:float=0.266511, resolution:float=0.07) -> pd.DataFrame:
    """Creates a pinned ellipse profile in 2D to simulate the extrusion profile. It is made in Y = 0 plane.\n
    Available format options:\n
    * oblong
    * ellipse
    * pinned ellipse
    * ellipse oblong 

    References:\n
    * [COMMINAL et al., 2018](https://doi.org/10.1016/j.addma.2017.12.013)\n
    * [Hebda et al., 2019](http://hdl.handle.net/10454/16895)\n
    * [Xu et al., 2022](https://hal-mines-paristech.archives-ouvertes.fr/hal-03766358)"""

    df = pd.DataFrame(columns=['x', 'y', 'z'])
    z_vector = np.linspace(0, height, int(height / resolution) + 2)

    # Draw contours right side
    if geometry == 'oblong':
        for z in z_vector:
            angle = np.arcsin((z - height / 2) * 2/ height)
            x = np.cos(angle) * height / 2 + (width - height) / 2
            new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
            df = pd.concat([df, new_row])

    elif geometry == 'ellipse':
        for z in z_vector:
            angle = np.arcsin((z - height / 2) * 2 / height)
            x = np.cos(angle) * width / 2
            new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
            df = pd.concat([df, new_row])

    elif geometry == 'pinned ellipse':
        x = 0
        for z in z_vector[::-1]:
            if x <= 0.95 * width / 2:  # Upper side (ellipse)
                angle = np.arcsin((z - height / 2) * 2 / height)
                x = np.cos(angle) * width / 2
                new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
                df = pd.concat([df, new_row])
            else:  # Lower side (rectangle)
                new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
                df = pd.concat([df, new_row])

    elif geometry == 'ellipse oblong':
        for z in z_vector:
            if z >= height / 2:  # Upper side (ellipse)
                angle = np.arcsin((z - height / 2) * 2 / height)
                x = np.cos(angle) * width / 2
                new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
                df = pd.concat([df, new_row])
            else:  # Lower side (oblong)
                angle = np.arcsin((z - height / 2) * 2 / height)
                x = np.cos(angle) * height/ 2 + (width - height) / 2
                new_row = pd.DataFrame([[x, 0, z]], columns=['x', 'y', 'z'])
                df = pd.concat([df, new_row])

    # Mirror to the left side
    df = pd.concat([df, df * [-1, 1, 1]])

    # Fill up
    for z in z_vector:
        x = df[df.z == z].x.to_list()
        x_vector = np.linspace(min(x), max(x), int((max(x) - min(x)) / resolution))[1:-1]
        z_row = np.ones_like(x_vector) * z
        new_row = new_row = pd.DataFrame.from_dict({'x':x_vector, 'y':np.zeros_like(x_vector), 'z':z_row})
        df = pd.concat([df, new_row])
    return df

def draw_rounded_ending(profile:pd.DataFrame, steps:int=7):
    """Draw a rounded ending by revoluting half of the profile in its Z axis"""
    ending = pd.DataFrame(columns=profile.columns)
    one_side_profile = profile[profile.x < 0]
    for angle in np.linspace(0, np.pi, steps)[1:-1]:
        r = R.from_euler('z', angle, False)
        rotated = one_side_profile.to_numpy() @ r.as_matrix()
        df_rotated = pd.DataFrame(rotated, columns=profile.columns)
        ending = pd.concat([ending, df_rotated])
    return ending

def show_profile(profile:pd.DataFrame, axis:tuple=['x', 'z']):
    """Plot the profile to check it graphically"""
    fig, ax = plt.subplots()
    ax.plot(profile[axis[0]], profile[axis[1]], 'ko')
    ax.set_box_aspect(1)
    ax.set_title('profile')
    plt.gca().set_aspect('equal')
    plt.show()

def draw_tubular_point_cloud(pt0:tuple[float], pt1:tuple[float], profile:pd.DataFrame, resolution:float):
    """Given two coordinates and a profile, returns a DataFrame representing a tubular point cloud between them.\n
    Args:\n
    - points = [x, y, z]\n
    - profile = pd.DataFrame(columns=['x', 'y', 'z'])
    """
    x0, y0, z0 = pt0
    x1, y1, z1 = pt1
    angle = -np.arctan2(y1 - y0, x1 - x0)
    r = R.from_euler('z', angle + np.pi/2, False).as_matrix()
    rotated = profile.to_numpy() @ r

    dist = np.linalg.norm(np.array(pt1) - np.array(pt0))
    segments = int(dist/resolution) + 2
    X = np.repeat(np.linspace(x0, x1, num=segments), len(rotated))
    Y = np.repeat(np.linspace(y0, y1, num=segments), len(rotated))
    Z = np.array([z0 for i in range(len(X))])

    pcl = np.array([X, Y, Z]).transpose() + np.tile(rotated, (segments, 1))
    df_pcl = pd.DataFrame(pcl, columns=['X', 'Y', 'Z'])

    # Add rounded endings to both sides of tubular point cloud
    ending_part = draw_rounded_ending(profile)
    ending = pd.DataFrame(ending_part.to_numpy() @ r + np.array(pt1), columns=df_pcl.columns)
    opposite_r = R.from_euler('z', angle - np.pi/2, False).as_matrix()
    beginning = pd.DataFrame(ending_part.to_numpy() @ opposite_r + np.array(pt0), columns=df_pcl.columns)
    df_pcl = pd.concat([df_pcl, beginning, ending])
    return df_pcl

def build_command(index:int, commands:pd.DataFrame, profile:pd.DataFrame, resolution:float):
    """Builds one line of gcode command"""
    start_row = commands.iloc[index - 1] if index > 0 else commands.iloc[0]
    start_point = [start_row['X-Value'], start_row['Y-Value'], start_row['Z-Value']]
    end_point = [commands.iloc[index]['X-Value'], commands.iloc[index]['Y-Value'], commands.iloc[index]['Z-Value']]
    pcl = draw_tubular_point_cloud(start_point, end_point, profile, resolution)
    return pcl

def build_part(commands:pd.DataFrame, profile:pd.DataFrame, resolution:float, processes:int=2):
    """Builds every line in gcode commands. Number of processes can be changed, but with caution."""
    with multiprocessing.Pool(processes) as pool:
        df_pcl = pd.concat(pool.starmap(build_command, zip(commands[commands.Extrusion > 0].index, repeat(commands), repeat(profile), repeat(resolution))))
    return df_pcl

# if __name__ == '__main__':
#     import generate_pointcloud as gp
#     import utils
#     import os
#     start = time.time()

#     gcode_path = utils.open_or_download('63847798504e7d63865e5769')
#     df = gcode2dataframe(gcode_path, create_pcl_for_each_layer=False)
#     resol = 0.07

#     W, H = calculate_profile_dimensions('hebda')
#     profile = draw_profile('oblong', W, H, resolution=resol)

#     pcl = build_part(df, profile, resol)
    
#     path_to_pcl = gp.generate_pcl(pcl, path="files/", name = "00_nominal_part", LLS ="")

#     log.info("I took {:.1f} s to generate realistic point cloud.".format(time.time() - start))