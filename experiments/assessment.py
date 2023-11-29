import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.contrib.itertools import product
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import seaborn as sns
sns.set()
import utils

log = utils.init_logging("DEBUG")

def load(results_path:str = "data/experiments/results/v2",
         temperature:int = 200,
        speed:int = 1000,
        layer:int = 2,
    ):
    # Load the point cloud
    pcl_path = os.path.join(results_path, f"temperature_{temperature}_speed_{speed}", f"{layer}_LLS2.npy")
    npy = np.load(pcl_path)

    # Invert z axis
    npy[:, 2] = -npy[:, 2] + max(npy[:, 2])

    # Crop to ROI
    roi = [[None, None], [0, None], [None, 10]]
    npy = utils.crop_to_roi(npy, roi=roi)
    return npy

def get_strand_rois() -> dict[list[list[float]]]:
    """Definition of the ROI of each strand.
    strand_rois[feeding_rate] = [[x_min, x_max], [y_min, y_max], [z_min, z_max]] """
    strands_rois = {0.03: [[-28, -25], [40, 80], [None, None]]}
    for i in range(1, 20):
        strands_rois[0.03 + i * 0.01] = [[strands_rois[0.03][0][0] + i * 3, strands_rois[0.03][0][1] + i *3], *strands_rois[0.03][1:]]
    return strands_rois

def calculate_profile(strand:np.ndarray, layer:int, plot:bool=True):
    # Noise removal
    # Firstly, we split the lower third of the strand into two: left and right parts
    # Then we calculate the gap between the left and right parts, add a secure margin of 10% to both ends and remove any points outside that range
    upper_threshold = min(strand[:, 2]) + (max(strand[:, 2]) - min(strand[:, 2])) * 2/3
    lower_threshold = min(strand[:, 2]) + (max(strand[:, 2]) - min(strand[:, 2])) * 1/3
    upper_piece = strand[strand[:, 2] > upper_threshold]
    lower_piece = strand[strand[:, 2] < lower_threshold]
    left_part = lower_piece[lower_piece[:, 0] < upper_piece[:, 0].mean()]  # The left part of the bed before the strand
    right_part = lower_piece[lower_piece[:, 0] > upper_piece[:, 0].mean()]  # The right part of the bed after the strand
    bed_z = lower_piece[:, 2].mean()  # The z coordinate of the bed

    # Calculate height
    top_10 = sorted(strand[:, 2])[-10:]  # Get the top 10 highest points
    height = np.mean(top_10) - bed_z
    x_height = strand[strand[:, 2].argmax(), 0]
    last_layer_height = height / (layer + 1) + bed_z
    if layer > 0: strand = strand[strand[:, 2] > last_layer_height]  # Exclude previous layers

    if len(left_part) > 0 and len(right_part) > 0:
        lower_gap = min(right_part[:, 0]) - max(left_part[:, 0])
        strand = strand[strand[:, 0] > (max(left_part[:, 0]) - lower_gap * 0.1)]  # Remove noise from the left side
        strand = strand[strand[:, 0] < (min(right_part[:, 0]) + lower_gap * 0.1)]  # Remove noise from the right side
        strand = strand[strand[:, 2] > bed_z]  # Remove noise from the bottom
        if len(strand) == 0:
            print("No strand left after noise removal")
            return (None, None, None) if plot else (None, None)
    else:
        print("No gap between the left and right parts of the bed")
        return (None, None, None) if plot else (None, None)

    # Calcultate bed
    # Assuming the data can be sliced in 3 parts by z axis, the upper part should represent the strand and the lower should represent the bed
    # Bed is then calculated as the mean of the lower part
    upper_threshold = min(strand[:, 2]) + (max(strand[:, 2]) - min(strand[:, 2])) * 2/3
    lower_threshold = min(strand[:, 2]) + (max(strand[:, 2]) - min(strand[:, 2])) * 1/3
    upper_piece = strand[strand[:, 2] > upper_threshold]
    middle_piece = strand[(strand[:, 2] > lower_threshold)]
    middle_piece = middle_piece[(middle_piece[:, 2] < upper_threshold)]
    lower_piece = strand[strand[:, 2] < lower_threshold]
    left_part = lower_piece[lower_piece[:, 0] < upper_piece[:, 0].mean()]  # The left part of the bed before the strand
    right_part = lower_piece[lower_piece[:, 0] > upper_piece[:, 0].mean()]  # The right part of the bed after the strand

    if len(lower_piece) == 0 or len(upper_piece) == 0:
        return (None, None, None) if plot else (None, None)

    # Calculate width
    # We calculate the width by three ways:
    #  * difference between opposite points in the middle part
    #  * difference between opposite points in the upper part
    #  * gap size in the center
    # We then exclude any values outside the standart deviation and the width will be the mean of the remaining values
    middle_left = middle_piece[middle_piece[:, 0] < upper_piece[:, 0].mean()]
    middle_right = middle_piece[middle_piece[:, 0] > upper_piece[:, 0].mean()]
    middle_width = np.mean(middle_right[:, 0]) - np.mean(middle_left[:, 0]) if len(middle_left) > 0 and len(middle_right) > 0 else None
    upper_width = max(upper_piece[:, 0]) - min(upper_piece[:, 0])
    lower_gap = min(right_part[:, 0]) - max(left_part[:, 0]) if len(left_part) > 0 and len(right_part) > 0 else None
    widths = np.array([width for width in [middle_width, upper_width, lower_gap] if width is not None])
    width = upper_width
    if len(widths) == 0:
        return None, None
    elif len(widths) == 1:
        width = widths[0]
    elif len(widths) >= 2:
        width = widths.mean()
        
    if plot:
        width_start = upper_piece[:, 0].mean() - width/2
        width_end = upper_piece[:, 0].mean() + width/2
        fig, ax = plt.subplots()
        ax.plot(strand[:, 0], strand[:, 2], 'ko')
        # ax.plot(xaxis, interp(xaxis), '-')
        ax.plot([min(strand[:, 0]), max(strand[:, 0])], [bed_z, bed_z], '-k')  # plot bed
        ax.plot([min(strand[:, 0]), max(strand[:, 0])], [upper_threshold, upper_threshold], '--k')  # plot upper threshold
        ax.plot([min(strand[:, 0]), max(strand[:, 0])], [lower_threshold, lower_threshold], '--k')  # plot lower threshold
        ax.plot([x_height, x_height], [bed_z, bed_z + height], '-g')  # plot height
        ax.plot([width_start, width_end], [np.mean([lower_threshold, upper_threshold]), np.mean([lower_threshold, upper_threshold])], '-r')  # plot width
        ax.legend(['measurements', 'bed', 'upper third', 'lower third', 'height', 'width'])
        ax.set_box_aspect(1)
        ax.set_title('Front view')
        plt.close()
        return fig, height, width
    
    return height, width

def assess_width_and_height(strand:np.ndarray, layer:int) -> dict:
    step = 0.2
    heights = []
    widths = []
    for y_slice in np.arange(min(strand[:, 1]), max(strand[:, 1]), step):
        # Cut a straight line
        slice = strand[strand[:, 1] >= y_slice]
        slice = slice[slice[:, 1] < (y_slice + step)]
        if len(slice) < 10: continue
        height, width = calculate_profile(slice, layer, False)
        if height is None or width is None: continue
        heights.append(height)
        widths.append(width)

    stats = pd.DataFrame({'height':heights, 'width':widths})
    
    return {'width':{'mean': stats.width.mean(), 'std': stats.width.std()}, 'height':{'mean': stats.height.mean(), 'std': stats.height.std()}}

def assess_results() -> pd.DataFrame:
    results = pd.DataFrame(columns=['temperature', 'speed', 'run_number', 'layer', 'feeding_rate', 'extrusion_type', 'width_mean', 'width_std', 'height_mean', 'height_std'])

    strands_rois = get_strand_rois()
    for temperature, speed, run_number, layer, feeding_rate, extrusion_type in product([180, 200], [500, 700, 1000, 1200], range(1, 4), range(3), strands_rois.keys(), ['one_line', 'overextrusion']):
        measurement = load(temperature=temperature, speed=speed, exp_number=run_number, layer=layer)
        roi = strands_rois[feeding_rate][extrusion_type]
        strand = utils.crop_to_roi(measurement, roi=roi)
        stats = assess_width_and_height(strand, layer)
        results.loc[len(results)] = {'temperature':temperature, 'speed':speed, 'run_number':run_number, 'layer':layer, 'feeding_rate':feeding_rate, 'extrusion_type': extrusion_type, 'width_mean':stats['width']['mean'], 'width_std':stats['width']['std'], 'height_mean':stats['height']['mean'], 'height_std':stats['height']['std']}
        
        # Clean up
        del measurement, strand, stats
    return results

def assess_width_and_height_by_regression(strand:np.ndarray, last_layer_height:float, plot:bool=True):
    # Model fitting
    X = strand[:, 0].reshape(-1, 1)
    Z = strand[:, 2].reshape(-1, 1)

    model = Pipeline([
                    ('spline', SplineTransformer(n_knots=50, degree=10)),
                    ('linear', LinearRegression())])
    model = model.fit(X, Z)
    pred = model.predict(X)
    r2 = r2_score(Z, pred)

    domain = np.linspace(min(strand[:, 0]), max(strand[:, 0]), 3000)
    pred_curve = model.predict(domain.reshape(-1, 1))
    prediction = np.column_stack([domain, pred_curve])

    # Noise removal
    prediction = prediction[prediction[:, 0] > np.percentile(prediction[:, 0], 5)]  # Remove noise from the left side
    prediction = prediction[prediction[:, 0] < np.percentile(prediction[:, 0], 95)]  # Remove noise from the left side
    center = prediction[prediction[:, 0] > np.median(X) - 0.5]
    center = center[center[:, 0] < np.median(X) + 0.5]
    top = center[center[:, 1] > np.percentile(center[:, 1], 85)]  # Top 15% highest points
    x_center = top[top[:, 1].argmax(), 0]
    lower_third = prediction[prediction[:, 1] < np.percentile(prediction[:, 1], 50)]
    left_side = lower_third[lower_third[:, 0] < x_center]
    right_side = lower_third[lower_third[:, 0] > x_center]
    lower_gap = right_side[:, 0].min() - left_side[:, 0].max()
    prediction = prediction[prediction[:, 0] > (left_side[:, 0].max() - lower_gap * 0.1)]  # Remove noise from the left side
    prediction = prediction[prediction[:, 0] < (right_side[:, 0].min() + lower_gap * 0.1)]  # Remove noise from the right side

    # Calculation of characteristics
    bed = np.median(lower_third[:, 1])
    basis = bed + last_layer_height

    # Height is the highest point minus the basis
    highest_point = prediction[prediction[:, 1].argmax()]
    raw_height = highest_point[1]
    height = raw_height - basis
    half_height = basis + height/2

    # Width is calculated at half height
    left = prediction[prediction[:, 0] < highest_point[0]]
    right = prediction[prediction[:, 0] > highest_point[0]]
    if len(left) == 0 or len(right) == 0:  # If the highest point is at the edge of the bed, we can't calculate the width
        return (None, None, None) if plot else (None, None)
    width_left = left[np.abs(left[:, 1] - half_height).argmin()]
    width_right = right[np.abs(right[:, 1] - half_height).argmin()]
    width = width_right[0] - width_left[0]

    # Area is calculated as the area under the curve minus the area under the basis
    basis_curve = prediction[:, 1].copy()
    basis_curve[basis_curve > basis] = basis
    basis_area = np.trapz(basis_curve, prediction[:, 0])
    area = np.trapz(prediction[:, 1], prediction[:, 0]) - basis_area

    if plot:
        fig, ax = plt.subplots()
        ax.plot(strand[:, 0], strand[:, 2], 'bo', markersize=0.02, label='measurement points')  # real data
        ax.plot([min(strand[:, 0]), max(strand[:, 0])], [bed, bed], '-k', label='bed')  # plot bed
        ax.plot([min(strand[:, 0]), max(strand[:, 0])], [bed + last_layer_height, bed + last_layer_height], '--k', label='last layer hight')  # plot last layer height
        ax.plot([highest_point[0], highest_point[0]], [basis, highest_point[1]], '-g', label='height')  # plot height
        ax.plot([width_left[0], width_right[0]], [half_height, half_height], '-m', label='width')  # plot width
        ax.fill_between(prediction[:, 0], np.ones_like(prediction[:, 0]) * bed, basis_curve, color='b', alpha=0.2, label='area')  # area
        ax.plot(prediction[:, 0], prediction[:, 1], 'r-', label='regression')  # regression
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_box_aspect(1)
        ax.set_title('Front view')
        plt.close()
        return fig, height, width, area, r2
    else:
        return height, width, area, r2
    
def assess_by_regression():
    results = pd.DataFrame(columns=['temperature', 'speed', 'layer', 'feeding_rate', 'width', 'height', 'area', 'determination'])

    strands_rois = get_strand_rois()
    for temperature, speed, feeding_rate in product([180, 200], [500, 1200], strands_rois.keys()):
        last_layer_height = 0
        for layer in range(3):
            measurement = load(temperature=temperature, speed=speed, layer=layer)
            roi = strands_rois[feeding_rate]
            strand = utils.crop_to_roi(measurement, roi=roi)
            try:
                fig, height, width, area, r2 = assess_width_and_height_by_regression(strand, last_layer_height=last_layer_height, plot=True)
            except Exception as e:
                print(f"Error while assessing {temperature}, {speed}, {layer}, {feeding_rate}")
                NameError(e)

            results.loc[len(results)] = {'temperature':temperature, 'speed':speed, 'layer':layer, 'feeding_rate':feeding_rate, 'width':width, 'height':height, 'area': area, 'determination':r2}
            fig.savefig(f"data/experiments/assessment/v2/img/{temperature}_{speed}_{layer}_{feeding_rate}.png")
            last_layer_height += height
    return results

if __name__ == '__main__':
    results = assess_by_regression()
    results.to_csv('data/experiments/assessment/v2/assessments_v1.csv', index=False)