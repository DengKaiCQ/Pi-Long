"""
Using evo version: 1.14.0

This script calls evo.tools.plot, which defaults to TkAgg backend 
and tries to open a GUI window for saving the figures
(even the code doesn't actually open a window).
On headless servers the figure saving will fail, but caculating ate is still functional.
"""

import numpy as np
import numpy
import os

from evo.core import sync
from evo.core.metrics import PoseRelation

import numpy as np

import numpy
import os
import argparse



def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Args:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

    Returns:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches

def align(model,data,calc_scale=True):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    if calc_scale:
        rotmodel = rot*model_zerocentered
        dots = 0.0
        norms = 0.0
        for column in range(data_zerocentered.shape[1]):
            dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
            normi = numpy.linalg.norm(model_zerocentered[:,column])
            norms += normi*normi
        # s = float(dots/norms)  
        s = float(norms/dots)
    else:
        s = 1.0  

    # trans = data.mean(1) - s*rot * model.mean(1)
    # model_aligned = s*rot * model + trans
    # alignment_error = model_aligned - data

    # scale the est to the gt, otherwise the ATE could be very small if the est scale is small
    trans = s*data.mean(1) - rot * model.mean(1)
    model_aligned = rot * model + trans
    data_alingned = s * data
    alignment_error = model_aligned - data_alingned
    
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error, s


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 

    Args:
        ax -- the plot
        stamps -- time stamps (1xn)
        traj -- trajectory (3xn)
        style -- line style
        color -- line color
        label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][2])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(first_list, second_list, plot="", use_alignment=False, _args=""):
    # parse command line
    parser = argparse.ArgumentParser(
        description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.')

    parser.add_argument(
        '--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument(
        '--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument(
        '--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    parser.add_argument(
        '--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations',
                        help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument(
        '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument(
        '--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args(_args)
    args.plot = plot

    matches = associate(first_list, second_list, float(
        args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")

    first_xyz = numpy.matrix(
        [[float(value) for value in first_list[a][0:3, 3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale)
                              for value in second_list[b][0:3, 3]] for a, b in matches]).transpose()

    if use_alignment:
        rot, trans, trans_error, s = align(second_xyz, first_xyz)
        print('scale:',s)
        second_xyz_aligned = (rot * second_xyz + trans)/s
    else:
        alignment_error = second_xyz - first_xyz
        trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
            alignment_error, alignment_error), 0)).A[0]
        second_xyz_aligned = second_xyz

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in first_list[b][0:3, 3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale)
                                   for value in second_list[b][0:3, 3]] for b in second_stamps]).transpose()
    if use_alignment:
        second_xyz_full_aligned = (rot * second_xyz_full + trans)/s
    else:
        second_xyz_full_aligned = second_xyz_full

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (
            a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write("\n".join(["%f " % stamp+" ".join(["%f" % d for d in line])
                   for stamp, line in zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()
    align_option = 'aligned' if use_alignment else 'no_align'

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error))
        ax.set_title(
            f'ate-rmse of {len(trans_error)} pose pairs ({align_option}):{float(ATE):0.4f}m')
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,
                  '--', "black", "ground truth")
        ax.plot(first_xyz_full.transpose().A[0][0], first_xyz_full.transpose(
        ).A[0][1], marker="o", markersize=5, markerfacecolor="green", label="start gt")
        ax.plot(first_xyz_full.transpose().A[-1][0], first_xyz_full.transpose(
        ).A[-1][1], marker="o", markersize=5, markerfacecolor="yellow", label="end gt")

        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose(
        ).A, '-', "blue", "estimated")
        ax.plot(second_xyz_full_aligned.transpose().A[0][0], second_xyz_full_aligned.transpose(
        ).A[0][1], marker="*", markersize=5, markerfacecolor="cyan", label="start estimated")
        ax.plot(second_xyz_full_aligned.transpose().A[-1][0], second_xyz_full_aligned.transpose(
        ).A[-1][1], marker="*", markersize=5, markerfacecolor="purple", label="end estimated")

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
            # ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
            label = ""
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot, dpi=300)

    return {
        "compared_pose_pairs": (len(trans_error)),
        "absolute_translational_error.rmse": numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)),
        "absolute_translational_error.mean": numpy.mean(trans_error),
        "absolute_translational_error.median": numpy.median(trans_error),
        "absolute_translational_error.std": numpy.std(trans_error),
        "absolute_translational_error.min": numpy.min(trans_error),
        "absolute_translational_error.max": numpy.max(trans_error),
    }


def evaluate(poses_gt, poses_est, plot, use_alignment=False):

    poses_gt = poses_gt
    poses_est = poses_est

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, plot, use_alignment=use_alignment)
    print(results)
    return results

def load_poses_monogs(pose_file, inv = False):
    """Load ground truth poses (T_w_cam0) from file."""
    poses = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        data = np.array([list(map(float, line.split())) for line in lines])

        for line in data:
            T_w_cam0 = line.reshape(4, 4)
            if inv:
                T_w_cam0 = np.linalg.inv(T_w_cam0)
            poses.append(T_w_cam0)

    return np.array(poses)

def load_poses_vkitti(pose_file, inv = True):
    """Load ground truth poses (T_w_cam0) from file."""
    poses = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        # lines = lines[::2]
        data = np.array([list(map(float, line.split())) for line in lines])

        for line in data:
            line = line[1:]
            T_w_cam0 = line.reshape(4, 4)
            if inv:
                T_w_cam0 = np.linalg.inv(T_w_cam0)
            poses.append(T_w_cam0)

    return np.array(poses)

import numpy as np
from scipy.spatial.transform import Rotation


def load_poses_dpvo(pose_file):
    """Load ground truth poses (T_w_cam0) from file."""
    poses = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()

        for line in lines:

            if line.startswith('#'):
                continue

            data = list(map(float, line.split()))

            time, tx, ty, tz, qw, qx, qy, qz = data

            T_w_cam0 = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw), tx],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw), ty],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2), tz],
                [0, 0, 0, 1]
            ])
            poses.append(T_w_cam0)

    return np.array(poses)


def load_poses_waymo(pose_dir, inverse=False):

    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    poses = []

    for file_name in pose_files:
        file_path = os.path.join(pose_dir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        matrix = []
        for line in lines:
            if line.startswith('#'):  
                continue
            row = list(map(float, line.strip().split()))
            if len(row) == 4:  
                matrix.append(row)
        
        if len(matrix) != 4:
            raise ValueError(f" {file_name} format error: need 4x4, but we get {len(matrix)} rows")
        
        pose = np.array(matrix, dtype=np.float64)
        
        
        if inverse:
            pose = np.linalg.inv(pose)
        
        poses.append(pose)
    
    return np.stack(poses, axis=0)  # (N, 4, 4)

def load_idx(idx_file):
    lis = []
    with open(idx_file, 'r') as f:
        lines = f.readlines()
        lis = [int(line) for line in lines]
    return lis

def load_poses_kitti(pose_file, index = False, inv = False):
    """Load ground truth poses (T_w_cam0) from file."""
    # pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

    # Read and parse the poses
    poses = []
    idx = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        data = np.array([list(map(float, line.split())) for line in lines])

        for line in data:
            if index == False:
                T_w_cam0 = line[:12].reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            else:
                T_w_cam0 = line[1:13].reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                idx.append(line[0])
            
            if inv == False:
                poses.append(T_w_cam0)
            else:
                poses.append(np.linalg.inv(T_w_cam0))
    if index == False:
        return np.array(poses)
    else:
        return np.array(poses), idx

def calculate_trajectory_length(poses):
    translations = poses[:, :3, 3]
    
    distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    
    total_length = np.sum(distances)
    
    return total_length

from scipy.spatial.transform import Rotation as R
import numpy as np


if __name__ == '__main__':

    pose_est_path = './camera_poses.txt'
    pose_gt_path = './00.txt' 

    pose_est_data = load_poses_kitti(pose_est_path)
    pose_gt_data = load_poses_kitti(pose_gt_path)
    # pose_gt_data = load_poses_vkitti(pose_gt_path)

    trajectory_length = calculate_trajectory_length(pose_gt_data)
    print(f"Total trajectory length: {trajectory_length} meters")

    directory_path = os.path.dirname(pose_est_path)

    print(pose_est_data.shape)
    print(pose_gt_data.shape)

    import matplotlib.pyplot as plt
    import numpy as np
    import evo.core.trajectory as et
    import evo.tools.file_interface as ef
    import evo.tools.plot as ep

    est_pose = pose_est_data 
    gt_pose = pose_gt_data

    # create timestamps
    timestamps_gt = np.linspace(0, 1, gt_pose.shape[0])
    timestamps_est = np.linspace(0, 1, est_pose.shape[0])

    # calculate the index for equidistant sampling
    num_samples = est_pose.shape[0]
    indices = np.linspace(0, gt_pose.shape[0] - 1, num_samples, dtype=int)

    # perform equidistant sampling in gt_pose
    gt_pose = gt_pose[indices]
    timestamps_gt = timestamps_gt[indices]

    traj_est = et.PoseTrajectory3D(poses_se3=est_pose, timestamps=timestamps_est)
    traj_gt = et.PoseTrajectory3D(poses_se3=gt_pose, timestamps=timestamps_gt)

    from evo.core import trajectory, sync, metrics
    from evo.tools import file_interface
    print("registering and aligning trajectories")
    traj_ref, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    traj_est.align(traj_ref, correct_scale=True)

    # Compute ATE
    from evo.core.metrics import APE, PoseRelation
    ape_metric = APE(PoseRelation.full_transformation)
    ape_metric.process_data((traj_gt, traj_est))
    ape_stats = ape_metric.get_all_statistics()

    print(f"ATE: {ape_stats['rmse']}")

    from evo.core.metrics import RPE, PoseRelation
    rpe_metric = RPE(PoseRelation.full_transformation)
    rpe_metric.process_data((traj_gt, traj_est))
    rpe_stats = rpe_metric.get_all_statistics()

    print(f"RPE: {rpe_stats['rmse']}")

    file_name = 'ATE.txt'
    file_path = os.path.join(directory_path, file_name)

    with open(file_path, 'w') as file:
        file.write(f"ATE: {ape_stats['rmse']}\n")
        file.write(f"RPE: {rpe_stats['rmse']}\n")
        file.write(f"Total length: {trajectory_length} m")

    print(f"ATE results has been writen to: {file_path}")

    print("loading plot modules")
    from evo.tools import plot
    import matplotlib.pyplot as plt
 
    print("plotting")

    fig_2 = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz
    ax = plot.prepare_axis(fig_2, plot_mode)
    ax.set_aspect('equal', 'box')
    ax.set_box_aspect(1)

    color_purple = '#602357'
    color_dark_red = '#B6443F'
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'GT Poses', linewidth=4.5)
    plot.traj(ax, plot_mode, traj_est, '-', color_dark_red, 'Estimated Poses', linewidth=4.5)

    pic_saving_path = os.path.join(directory_path, "trajectory_plot.pdf")
    pic_saving_path = os.path.join(directory_path, "trajectory_plot.png")
    fig_2.savefig(pic_saving_path)
    pic_saving_path = os.path.join(directory_path, "trajectory_plot.svg")
    fig_2.savefig(pic_saving_path)

