import logging
from pprint import pformat
import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import expm, norm
import scipy.stats as stats
import os
import plotly.graph_objs as go
from .utils import bundle_adjust
import peakutils as peakutils
from skimage.transform import (hough_line, hough_line_peaks)
from skimage.feature import canny
from skimage.morphology import opening, closing
from sklearn.cluster import KMeans
import open3d as o3d

# TODO: recycle this code into a helpers module or class

log = logging.getLogger('SmoPa3D')

def histogram1d(data, bins='auto'):
    n, bins, patches = plt.hist(data, bins, facecolor='blue', alpha=0.5, density=True)
    return n, bins, patches


def segmentation(heatmap, threshold):
    return np.where(heatmap > threshold, 1.0, 0.0)


def to_grayscale(heatmap):
    maxGray = max(heatmap.flatten())
    return np.uint8(heatmap / maxGray * 255)


def bin_centers(binedges):
    return [(binedges[idx]+binedges[idx+1])/2 for idx in range(len(binedges[:-1]))] 


def cluster(points, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return cluster_centers, labels


def to_parameterform(angle, distance):
    x0 = distance * np.cos(angle)
    y0 = distance * np.sin(angle)
    x1 = x0 - distance * np.sin(angle)
    y1 = y0 + distance * np.cos(angle)
    return x0, x1, y0, y1


def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return [px, py]


def remove_outliers(intersections, centers, labels, min_Npoints=3):
    # remove outlier (clusters with less than min_Npoints in the cluster intersections)
    sorted_labels = labels.argsort()
    labels = labels[sorted_labels]
    unique, counts = np.unique(labels, return_counts=True)
    occurences = dict(zip(unique, counts))

    bad_clusters = [key for key, val in occurences.items() if val < min_Npoints]

    for bad_cluster in bad_clusters:
        bad_cluster_idx = np.nonzero(labels == bad_cluster)
        labels = np.delete(labels, bad_cluster_idx)
        intersections = np.delete(intersections, bad_cluster_idx, axis=0)
    centers = np.delete(centers, bad_clusters, axis=0)
    return intersections, centers, labels


def apply_hough_transform(heatmap, useEdges=True):
    grayscale = opening(heatmap.T)
    if useEdges:
        grayscale = canny(grayscale/255.)
        grayscale = closing(grayscale, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        grayscale = np.where(grayscale != 0.0, 255, 0)
    h, theta, d = hough_line(grayscale)
    return grayscale, h, theta, d


def find_intersections(angles, distances, xBins, yBins, num_xbins, num_ybins):
    intX = []
    intY = []
    for angle, dist in zip(angles, distances):
        x1, x2, y1, y2 = to_parameterform(angle, dist)
        for angle2, dist2 in zip(angles, distances):
            x0 = 0
            y0 = 0
            if angle != angle2 and dist != dist2:
                x3, x4, y3, y4 = to_parameterform(angle2, dist2)
                X, Y = intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                xIntersect = X
                yIntersect = Y
                intX.append(xIntersect)
                intY.append(yIntersect)
            else:
                continue
    return np.array([intX, intY]).T


class LaserscannerSystem:
    """
    Class representing the system of two laserscanners.

    Functionality:
        - trigger the data acquisition
        - extract the pointClouds from the file given by the laser sensors
    """

    def __init__(self, pcl1, pcl2):
        self.cloud1 = self.format_cloud(pcl1)
        self.cloud2 = self.format_cloud(pcl2)
        self.cloud1.name = "PCL1"
        self.cloud2.name = "PCL2"
        self.merged_cloud = self.join_clouds()
        self.calibration_parameters = None

    def add_calibration_parameters(self, filename):
        import json
        with open(filename, 'r') as json_file:
            json_data = json_file.read()
            self.calibration_parameters = json.loads(json_data)

    def calibrate_pcls(self):
        if self.calibration_parameters is not None:
            log.info("Starting calibration")

            self.merged_cloud.projection2D('x', 'y', suffix='before_calibration')
            self.merged_cloud.projection2D('x', 'z', suffix='before_calibration')
            self.merged_cloud.projection2D('y', 'z', suffix='before_calibration')

            self.cloud1.shift('z', float(self.calibration_parameters['delta1_z']))
            self.cloud2.shift('z', float(self.calibration_parameters['delta2_z']))

            self.cloud1.rotate('x', 'z', float(self.calibration_parameters['theta1_xz']))
            self.cloud1.rotate('y', 'z', float(self.calibration_parameters['theta1_yz']))
            self.cloud2.rotate('x', 'z', float(self.calibration_parameters['theta2_xz']))
            self.cloud2.rotate('y', 'z', float(self.calibration_parameters['theta2_yz']))

            self.cloud1.shift('x', float(self.calibration_parameters['delta1_x']))
            self.cloud1.shift('y', float(self.calibration_parameters['delta1_y']))
            self.cloud1.rotate('x', 'y', float(self.calibration_parameters['theta1_xy']))

            self.cloud2.shift('x', float(self.calibration_parameters['delta2_x']))
            self.cloud2.shift('y', float(self.calibration_parameters['delta2_y']))
            self.cloud2.rotate('x', 'y', float(self.calibration_parameters['theta2_xy'])-np.pi)

            self.cloud1.update()
            self.cloud2.update()

            self.merged_cloud = self.join_clouds()
            self.merged_cloud.show3DCloud(points=157, suffix="beforeCalib")

            self.cloud2.move_rotate(np.array(self.calibration_parameters['R'])*0, np.array(self.calibration_parameters['T']))

            self.merged_cloud = self.join_clouds()
            self.merged_cloud.show3DCloud(points=157, suffix="afterCalib")
            self.merged_cloud.projection2D('x', 'y', suffix='after_calibration')
            self.merged_cloud.projection2D('x', 'z', suffix='after_calibration')
            self.merged_cloud.projection2D('y', 'z', suffix='after_calibration')

    def format_cloud(self, cloud):
        return PointCloud(cloud)

    def join_clouds(self):
        return PointCloud(np.concatenate([self.cloud1._data, self.cloud2._data]), name="MergedCloud")

    def calibrate(self):
        calibration = Calibration(self.cloud1, self.cloud2)

        # normalize to the minimum in z direction
        print("Normalize to minimum")
        calibration.normalize_to_minimum('z')

        # correct the tilt in z direction
        print("Correct the tilt in z direction")
        calibration.z_leveling()
        
        # shift both point clouds to bed level. This should be where the highest point density is found
        print("Bed Leveling")
        calibration.bed_leveling()
        self.cloud1.clean('z', 0.5, 1)
        self.cloud1.clean('y', 20, 100)
        self.cloud1.clean('x', 10, 40)
        self.cloud2.clean('z', 0.5, 1)

        self.merged_cloud = self.join_clouds()

        # for the calibration the assumption is, that only very narrow parts are printed
        # any noisy artifacts should be removed

        self.cloud1.clean('z', -2, 1)
        self.cloud2.clean('z', -2, 1)


        # extract a region around the floor with a minimum height of 0.5mm for the calibration in the x-y plane
        subcloud1 = self.cloud1.extract_subcloud('z', 0.5, 1)
        subcloud1.name = "Sub1"
        subcloud2 = self.cloud2.extract_subcloud('z', 0.5, 1)
        subcloud2.name = "Sub2"

        # find the origin of the calibration structure by clustering of heap points of hough lines
        angle_xy1, center1 = Calibration.calibrateXY(subcloud1, num_xbins=300, num_ybins=300, useEdges=False, min_distance=3)
        angle_xy2, center2 = Calibration.calibrateXY(subcloud2, num_xbins=300, num_ybins=300, useEdges=False, min_distance=2, min_Npoints=3)

        self.cloud1.shift('x', center1[0])
        self.cloud1.shift('y', center1[1])

        self.cloud2.shift('x', center2[0])
        self.cloud2.shift('y', center2[1])

        self.cloud1.rotate('x', 'y', angle_xy1)
        self.cloud2.rotate('x', 'y', angle_xy2+np.pi)

        R, T = calibration.RBFMerge()

        calibration.calibration_parameters['theta1_xy'] = angle_xy1
        calibration.calibration_parameters['theta2_xy'] = angle_xy2
        calibration.calibration_parameters['delta1_x'] = center1[0]
        calibration.calibration_parameters['delta1_y'] = center1[1]

        calibration.calibration_parameters['delta2_x'] = center2[0]
        calibration.calibration_parameters['delta2_y'] = center2[1]

        calibration.calibration_parameters['R'] = R.tolist()
        calibration.calibration_parameters['T'] = T.tolist()

        self.cloud2.move_rotate(R, T)

        calibration.save_parameters_to_file("data/Calibration_Parameters.txt")


class Calibration:
    """
    Class to takes two point clouds and performs processing steps to determine
    the 3 rotational angles and the three shifts between two point clouds

    Parameters:
        - 2 Point cloud objects (probably numpy arrays)
        - Calibration parameters

    Functionality:
        - Perform normalizations
        - Save and visualize the Point clouds
        - calculate the angles and shifts
        - save/ return the calibration parameters
    """
    def __init__(self, cloud1, cloud2):
        self.cloud1 = cloud1
        self.cloud2 = cloud2
        self.suffix = ""
        self.calibration_parameters =  {'delta1_x': None, 'delta2_x': None, 'delta1_y': None, 'delta2_y': None, 'delta1_z': None, 'delta2_z': None, 'theta1_xy': None, 'theta2_xy': None, 'theta1_xz': None, 'theta2_xz': None, 'theta1_yz': None, 'theta2_yz': None, 'R': None, 'T': None}

    def save_parameters_to_file(self, filename):
        import json
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        with open(filename, 'w') as json_file:
            dict_json = json.dump(self.calibration_parameters, json_file)

    def normalize_to_minimum(self, dim):
        min1 = min(self.cloud1.dim[dim])
        min2 = min(self.cloud2.dim[dim])
        self.cloud1.dim[dim] -= min1
        self.cloud2.dim[dim] -= min2
        self.suffix = ''.join([self.suffix, 'NZ'])

        if self.calibration_parameters['delta1_z'] is None:
            self.calibration_parameters['delta1_z'] = min1
        else:
            self.calibration_parameters['delta1_z'] += min1

        if self.calibration_parameters['delta2_z'] is None:
            self.calibration_parameters['delta2_z'] = min2
        else:
            self.calibration_parameters['delta2_z'] += min2

        self.cloud1.update()
        self.cloud2.update()

    def z_leveling(self):
        self.suffix = ''.join([self.suffix, 'ZL'])

        num_xbins = 201
        num_ybins = 801
        self.cloud1.projection2D('x', 'z', suffix="before_zleveled")

        ##log.info("Requiring threshold for segmentation")
        ##threshold = float(input("Enter threshold value according to control plot: "))
        theta_xz, intercept = self.get_rotational_angle(self.cloud1, 'x', 'z', num_xbins, num_ybins, suffix=self.suffix, name=self.cloud1.name)
        self.cloud1.rotate('x', 'z', theta_xz, 0, intercept)
        self.cloud1.correct_projection('x', theta_xz, min(self.cloud1.dim['x']))
        self.cloud1.dim['z'] -= intercept

        self.calibration_parameters['theta1_xz'] = theta_xz
        if self.calibration_parameters['delta1_z'] is None:
            self.calibration_parameters['delta1_z'] = intercept
        else:
            self.calibration_parameters['delta1_z'] += intercept
        self.cloud1.update()

        self.cloud1.projection2D('x', 'z', suffix="after_zleveled")
        self.cloud1.projection2D('y', 'z', suffix="before_zleveled")

        ##log.info("Requiring threshold for segmentation")
        ##threshold = float(input("Enter threshold value according to control plot: "))
        theta_yz, intercept = self.get_rotational_angle(self.cloud1, 'y', 'z', num_xbins, num_ybins, suffix=self.suffix, name=self.cloud1.name)
        self.cloud1.rotate('y', 'z', theta_yz, 0, intercept)
        self.cloud1.correct_projection('y', theta_yz, min(self.cloud1.dim['y']))

        self.cloud1.dim['z'] -= intercept
        self.calibration_parameters['theta1_yz'] = theta_yz
        if self.calibration_parameters['delta1_z'] is None:
            self.calibration_parameters['delta1_z'] = intercept
        else:
            self.calibration_parameters['delta1_z'] += intercept
        self.cloud1.update()

        self.cloud1.projection2D('y', 'z', suffix="after_zleveled")
        self.cloud2.projection2D('x', 'z', suffix="before_zleveled")


        ##log.info("Requiring threshold for segmentation")
        ##threshold = float(input("Enter threshold value according to control plot: "))
        theta_xz, intercept = self.get_rotational_angle(self.cloud2, 'x', 'z', num_xbins, num_ybins, suffix=self.suffix, name=self.cloud2.name)
        self.cloud2.rotate('x', 'z', theta_xz, 0, intercept)
        self.cloud2.correct_projection('x', theta_xz, min(self.cloud2.dim['x']))

        self.cloud2.dim['z'] -= intercept
        self.calibration_parameters['theta2_xz'] = theta_xz
        if self.calibration_parameters['delta2_z'] is None:
            self.calibration_parameters['delta2_z'] = intercept
        else:
            self.calibration_parameters['delta2_z'] += intercept
        self.cloud2.update()

        self.cloud2.projection2D('x', 'z', suffix="after_zleveled")
        self.cloud2.projection2D('y', 'z', suffix="before_zleveled")

        ##log.info("Requiring threshold for segmentation")
        ##threshold = float(input("Enter threshold value according to control plot: "))
        theta_yz, intercept = self.get_rotational_angle(self.cloud2, 'y', 'z', num_xbins, num_ybins, suffix=self.suffix, name=self.cloud2.name)
        self.cloud2.rotate('y', 'z', theta_yz, 0, intercept)
        self.cloud2.correct_projection('y', theta_yz, min(self.cloud2.dim['y']))    
        self.cloud2.dim['z'] -= intercept
        self.calibration_parameters['theta2_yz'] = theta_yz
        if self.calibration_parameters['delta2_z'] is None:
            self.calibration_parameters['delta2_z'] = intercept
        else:
            self.calibration_parameters['delta2_z'] += intercept
        self.cloud2.update()

        ##self.cloud2.projection2D('y', 'z', suffix="after_zleveled")


    def bed_leveling(self):
        self.cloud1.normalize_hist_to_maximum('z', 800)
        self.cloud2.normalize_hist_to_maximum('z', 800)
        self.cloud1.update()
        self.cloud2.update()

    def get_rotational_angle(self, cloud, dim1, dim2, num_xbins, num_ybins, xlims=None, ylims=None, suffix=None, name=None):
        # segmentation
        
        ybins = None
        xbins = None
        """
        ##grayscale, xbins, ybins = cloud.projection2D(dim1, dim2, num_xbins=num_xbins, num_ybins=num_ybins, suffix=suffix, xlims=None, ylims=None)
        cleaned_grayscale = np.where(grayscale > threshold, 1.0, 0.0)
        ##extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

        # get the relevant points for the fit
        x_Reg = np.repeat(xbins[:-1], cleaned_grayscale.shape[1])
        y_Reg = np.tile(ybins[:-1], cleaned_grayscale.shape[0])*cleaned_grayscale.flatten()

        relevant_points = np.where(cleaned_grayscale.flatten() != 0.0, True, False)

        x_Reg = x_Reg[relevant_points]
        y_Reg = y_Reg[relevant_points]
        """
        x_Reg = cloud.dim[dim1]
        y_Reg = cloud.dim[dim2]

        if ybins is None:
            ybins = np.linspace(min(y_Reg), max(y_Reg), num_ybins)
        if xbins is None:
            xbins = np.linspace(min(x_Reg), max(x_Reg), num_xbins)

        # Do the fit
        if np.all(x_Reg == x_Reg[0]):
            slope= 0
            intercept = 0
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_Reg, y_Reg)
        

        def line(x, slope, intercept):
            return intercept + slope*x
        y_Fit = line(xbins, slope, intercept)

        # Now correct the full data and return a control plot
        angle = np.arctan(slope)
        ##log.info("angle: {ANGLE}".format(ANGLE=angle))
        """
        plt.plot(x_Reg, y_Reg, 'o', label='original data')
        plt.plot(x_Reg, intercept + slope*x_Reg, 'r', label='fitted line')
        plt.legend()
        plt.show()
        """

        return angle, intercept

    @staticmethod
    def calibrateXY(cloud, num_xbins=300, num_ybins=300, suffix="", min_distance=2, useEdges=True, min_Npoints=5, chosenCenter=None):
        find_centers = True
        get_angle = True
        print("Find centers for shifting")
        while find_centers or get_angle:
            
            xbins = np.linspace(min(cloud.dim['x'])-10, max(cloud.dim['x'])+10, num_xbins)
            ybins = np.linspace(min(cloud.dim['y'])-10, max(cloud.dim['y'])+10, num_ybins)
            grayscale, xbins, ybins = cloud.projection2D('x', 'y', xbins=xbins, ybins=ybins, num_xbins=num_xbins, num_ybins=num_ybins, suffix=suffix, xlims=None, ylims=None)

            # calculate scale to turn values back into mm
            min_y = min(ybins)
            min_x = min(xbins)
            m_y = (max(ybins) - min(ybins))/num_ybins
            m_x = (max(xbins) - min(xbins))/num_xbins

            grayscale = to_grayscale(grayscale)

            xCenters = bin_centers(xbins)
            yCenters = bin_centers(ybins)

            extent = [xCenters[0], xCenters[-1], yCenters[0], yCenters[-1]]

            useEdges = input("Apply hough transform to edges? Y/N: ")
            if useEdges == "Y":
                useEdges = True
            else:
                useEdges = False
            
            min_distance = int(input("Choose a minimum distance (Default: {MIN_DISTANCE}): ".format(MIN_DISTANCE=min_distance)))
            
            grayscale, h, theta, d = apply_hough_transform(grayscale, useEdges=useEdges)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))    
            ax = axes.ravel()
            ax[0].imshow(grayscale, cmap=cm.viridis, origin='lower')
            ax[0].set_title('Input')
            ax[1].imshow(grayscale, cmap=cm.viridis, origin='lower')

            angles = []
            angles_mm = []
            distances = []
            intersections = []
            hough_lines = {'x0': [], 'x1': []}
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=min_distance)):
                if angle < 0:
                    angle += 2*np.pi

                angles.append(angle)
                distances.append(dist)
                y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1 = (dist - grayscale.shape[1] * np.cos(angle)) / np.sin(angle)
                x0 = 0
                x1 = grayscale.shape[1]
                hough_lines['x0'].append((m_x*x0+min_x, m_y*y0+min_y))
                hough_lines['x1'].append((m_x*x1+min_x, m_y*y1+min_y))
                ax[1].plot((x0, x1), (y0, y1), '-r', zorder=1)

            ax[1].set_xlim(-10, num_xbins+10)
            ax[1].set_ylim(-10, num_ybins+10)
            ax[1].set_title('Hough_lines')

            intersections = find_intersections(angles, distances, xCenters, yCenters, num_xbins, num_ybins)

            centers, labels = cluster(intersections)
            min_Npoints = int(input("Choose a minimum intersection points to find suited clusters (Default: {MIN_POINTS}): ".format(MIN_POINTS=min_Npoints)))

            intersections, centers, labels = remove_outliers(intersections, centers, labels, min_Npoints=min_Npoints)

            for idx, center in enumerate(centers):
                ax[1].scatter(center[0], center[1], marker="o", color="g", zorder=11)
                ax[1].text(center[0]+5, center[1]+5, str(idx), color="g")
            if not os.path.exists("figures"):
                os.mkdir("figures")
                
            plt.savefig("figures/{NAME}{SUFFIX}_houghlines_x_y.png".format(NAME=cloud.name,  SUFFIX="_{SUFFIX}".format(SUFFIX=suffix if suffix is not None else "")), format="png")

            fig, ax2 = plt.subplots(1, 1, figsize=(15, 10))
            ax2.imshow(grayscale, cmap=cm.viridis, origin='lower')
            ax2.scatter(intersections[:, 0], intersections[:, 1], marker="+", color="b", zorder=10, s=50)
            ax2.set_xlim(-10, num_xbins)
            ax2.set_ylim(-10, num_ybins)
            for idx, center in enumerate(centers):
                ax2.scatter(center[0], center[1], marker="o", color="g", zorder=11)
                ax2.text(center[0]+5, center[1]+5, str(idx), color="g")
            plt.tight_layout()
            plt.savefig("figures/{NAME}{SUFFIX}_intersections_x_y.png".format(NAME=cloud.name,  SUFFIX="_{SUFFIX}".format(SUFFIX=suffix if suffix is not None else "")), format="png")
            
            find_centers_input = input("Centers and hough lines, ok? Y/N: ")
            if (find_centers_input == 'Y' and find_centers):
                find_centers = False
                # transform from bins to mm
                centers[:, 0] *= m_x
                centers[:, 0] += min_x
                centers[:, 1] *= m_y
                centers[:, 1] += min_y

                # Let user decide which center to take and shift data accordingly
                log.info(pformat(centers))
                id_center = int(input("Enter index to use for coordinate origin for cloud: "))
                origin = centers[id_center]
                log.info("{CENTER} was chosen".format(CENTER=origin))

                log.info("---  shift original cloud to new origin")
                cloud.shift('x', origin[0])
                cloud.shift('y', origin[1])
                suffix += "C"
                log.info("--- done shifting to new origin")

            elif (find_centers_input == 'Y' and not find_centers):
                        # Take the endpoint of one hough line to get the rotational angle for align the two clouds
                get_angle = False
                print(pformat(hough_lines['x0']))
                id_hline = int(input("Enter index to use for line end for rotating subcloud1: "))
                hline = hough_lines['x0'][id_hline]
                print("{LINE} was chosen".format(LINE=hline))

                x, y = hline
                angle_xy = math.atan2(y, x)
                # cloud.rotate('x', 'y', angle_xy)
                cloud.projection2D('x', 'y', suffix="sdf")

            else:
                log.info("Repeat calculation of hough lines and clustering")
                find_centers = True

        return angle_xy, origin

    def RBFMerge(self):
        top_cloud1 = self.cloud1.extract_subcloud('z', 0.1, 5)
        top_cloud2 = self.cloud2.extract_subcloud('z', 0.1, 5)
        data_top_cloud1 = top_cloud1._data
        data_top_cloud2 = top_cloud2._data

        # Take random samples because cloud sizes have to be equal (n of entries)
        size = min([data_top_cloud1.shape[0], data_top_cloud2.shape[0]])
        print(size)

        rand_data_cloud1 = data_top_cloud1  #np.empty(shape=(size,3))
        rand_data_cloud2 = data_top_cloud2  # np.empty(shape=(size,3))
        # for i in range(0, size):
        #     rand_data_cloud1[i] = random.choice(data_top_cloud1)
        #     rand_data_cloud2[i] = random.choice(data_top_cloud2)

        # Points have to be in order for algorithm, norm allows sorting
        sort_rand_cloud1 = np.append(rand_data_cloud1, np.empty(shape=(rand_data_cloud1.shape[0],1)), axis = 1)
        sort_rand_cloud2 = np.append(rand_data_cloud2, np.empty(shape=(rand_data_cloud2.shape[0],1)), axis = 1)
        for i in range(0, sort_rand_cloud1.shape[0]):
            sort_rand_cloud1[i,3] = np.linalg.norm(sort_rand_cloud1[i,0:2])
        for i in range(0, sort_rand_cloud2.shape[0]):
            sort_rand_cloud2[i,3] = np.linalg.norm(sort_rand_cloud2[i,0:2])

        # Conversion to tuples only to allow sorting
        dtype_cloud_tuples = [('x', float), ('y', float), ('z', float), ('norm', float)]
        tuples_sort_cloud1 = np.empty(sort_rand_cloud1.shape[0], dtype=dtype_cloud_tuples)
        tuples_sort_cloud2 = np.empty(sort_rand_cloud2.shape[0], dtype=dtype_cloud_tuples)
        for i in range(0, tuples_sort_cloud1.shape[0]):
            tuples_sort_cloud1[i] = (sort_rand_cloud1[i,0],sort_rand_cloud1[i,1],\
                sort_rand_cloud1[i,2],sort_rand_cloud1[i,3])
        for i in range(0, tuples_sort_cloud2.shape[0]):
            tuples_sort_cloud2[i] = (sort_rand_cloud2[i,0],sort_rand_cloud2[i,1],\
                sort_rand_cloud2[i,2],sort_rand_cloud2[i,3])
            
        sorted_cloud1 = np.array(tuples_sort_cloud1,dtype=dtype_cloud_tuples)
        sorted_cloud2 = np.array(tuples_sort_cloud2,dtype=dtype_cloud_tuples)
        sorted_cloud1.sort(order='norm')
        sorted_cloud2.sort(order='norm')

        # Restore original arrays to allow algorithm to work
        cloud1_final = np.empty((sorted_cloud1.shape[0],3))
        cloud1_final[:,0] = sorted_cloud1['x']
        cloud1_final[:,1] = sorted_cloud1['y']
        cloud1_final[:,2] = sorted_cloud1['z']


        cloud2_final = np.empty((sorted_cloud2.shape[0],3))
        cloud2_final[:,0] = sorted_cloud2['x']
        cloud2_final[:,1] = sorted_cloud2['y']
        cloud2_final[:,2] = sorted_cloud2['z']

        R, T, adjusted = bundle_adjust(cloud1_final[:size], cloud2_final[:size])

        return R, T


class PointCloud:
    """
    Class for PointCloud data

    Functionality:
        - Load point cloud
        - Perform a rotation
        - merge with another point cloud
        - show point cloud
        - save it
    """

    def __init__(self, point_cloud, name=""):
        # point_cloud should be a numpy array with 3 columns and N entries.
        self._data = point_cloud
        self._x = self._data[:, 0]
        self._y = self._data[:, 1]
        self._z = self._data[:, 2]
        self.dim = {'x': self._x, 'y': self._y, 'z': self._z}
        self.name = name

    def update(self):
        self._x = self.dim['x']
        self._y = self.dim['y']
        self._z = self.dim['z']
        self._data = np.array([self._x, self._y, self._z]).transpose()

    def prepareScatter(self, points, color):
        return go.Scatter3d(x=(self._x[::points]), y=(self._y[::points]), z=(self._z[::points]), mode='markers', marker=dict(size=3, color=self._z[::points], colorscale='Viridis', opacity=0.8))

    
    def projection2D(self, dim1, dim2, xbins=None, ybins=None, num_xbins=200, num_ybins=200, suffix=None, xlims=None, ylims=None):
        # Build 2D projection
        x = self.dim[dim1]
        y = self.dim[dim2]
        if ybins is None:
            ybins = np.linspace(min(y), max(y), num_ybins)
        if xbins is None:
            xbins = np.linspace(min(x), max(x), num_xbins)

        ##fig, ax = plt.subplots()
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], density=True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        """
        im = ax.imshow(heatmap.T, extent=extent, origin='lower')
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        fig.colorbar(im)
        ax.axis('equal')
        if not os.path.exists("figures"):
            os.mkdir("figures")
        plt.savefig("figures/{NAME}{SUFFIX}_2Dprojection_{X}_{Y}.png".format(NAME=self.name, SUFFIX="_{SUFFIX}".format(SUFFIX=suffix if suffix is not None else ""), X=dim1, Y=dim2), format="png")
        plt.show()
        """
        return heatmap, xedges, yedges

    def rotation_matrix(self, axis, angle):
        return expm(np.cross(np.eye(3), axis/norm(axis)*angle))
    
    def rotate(self, dim1, dim2, angle, pointx=0, pointy=0):
        xData = self.dim[dim1]
        yData = self.dim[dim2]
        xDataPrime = pointx + np.cos(angle)*(xData - pointx) + np.sin(angle)*(yData - pointy)
        yDataPrime = pointy - np.sin(angle)*(xData - pointx) + np.cos(angle)*(yData - pointy)
        self.dim[dim1] = xDataPrime
        self.dim[dim2] = yDataPrime
        self.update()

    def move_rotate(self, R, T):
        temp = numpy.matmul(R, self._data.T).T + T
        self.dim['x'] = temp[:,0]
        self.dim['y'] = temp[:,1]
        self.dim['z'] = temp[:,2]
        self.update()


    def skew(self, dim1, dim2, k):
        xData = self.dim[dim1]
        yData = self.dim[dim2]

        xDataPrime = xData + k*yData 
        yDataPrime = yData 
        self.dim[dim1] = xDataPrime
        self.dim[dim2] = yDataPrime
        self.update()


    def correct_projection(self, dim, angle, reference_point = 0):
        """
        Scales the axis to take into account the effect of a tilted sesnor projecting the height profile on a plane sensor 
        """
        data = self.dim[dim]
        data -= reference_point
        data /= np.cos(angle)
        data += reference_point 
        self.dim[dim] = data
        self.update()


    def normalize_hist_to_maximum(self, dim, num_bins, threshold=0.5, min_dist=100):
        data = self.dim[dim]
        count, bins, patches = histogram1d(data, bins=int(num_bins))
        indexes = peakutils.indexes(count, thres=threshold, min_dist=min_dist)
        maxzidx = np.argmax(count[indexes])
        maximum = bins[indexes][maxzidx]
        self.shift(dim, maximum)

    def extract_subcloud(self, dim, lvalue=-999, hvalue=999):
        # extracts a part of the the cloud given the
        subcloud = PointCloud(self._data[np.where(self.dim[dim] > lvalue, True, False)])
        subcloud = subcloud._data[np.where(subcloud.dim[dim] < hvalue, True, False)]
        return PointCloud(subcloud)

    def clean(self, dim, lvalue=-999, hvalue=999):
        hdata = PointCloud(self._data[np.where(self.dim[dim] > lvalue, True, False)])
        ldata = PointCloud(hdata._data[np.where(hdata.dim[dim] < hvalue, True, False)])
        self._data = ldata._data
        self._x = self._data[:, 0]
        self._y = self._data[:, 1]
        self._z = self._data[:, 2]
        self.dim = {'x': self._x, 'y': self._y, 'z': self._z}
        return PointCloud(ldata._data)

    def shift(self, dim, value):
        self.dim[dim] -= value
        self.update()

    def save_to_npz(self, outname, *args, **kwargs):
        np.savez(outname, self._data, *args, **kwargs)

    def transform_to_np(self):
        x = self._data[:, 0]
        y = self._data[:, 1]
        z = self._data[:, 2]
        nparray = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)), axis = 1)

        return nparray

    def show3DCloud(self):
        npy = self.transform_to_np()
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(npy)
        o3d.visualization.draw_geometries([pcl])

