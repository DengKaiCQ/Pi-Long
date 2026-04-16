"""
自动化ATE/RPE评估脚本 - 支持多个实验根目录，输出汇总Excel表格
ATE统一使用RMSE（米）
"""

import numpy as np
import os
import re
import matplotlib.pyplot as plt
import evo.core.trajectory as et
from evo.core import sync
from evo.core.metrics import APE, RPE, PoseRelation
from evo.tools import plot
from datetime import datetime
import pandas as pd

# ==================== 在这里修改参数 ====================

# 多个实验根目录列表（每个目录包含带前缀的子目录，如 _kitti_..._00_image_2）
EXP_ROOTS = [
    'Pi3-long/all_poses/exps_pi3_se3_30',
    'Pi3-long/all_poses/exps_pi3_se3_60',
    'Pi3-long/all_poses/exps_pi3_se3_90',
    'Pi3-long/all_poses/exps_pi3_se3_120',

    'Pi3-long/all_poses/exps_pi3_se3_30_900px',
    'Pi3-long/all_poses/exps_pi3_se3_60_900px',
    'Pi3-long/all_poses/exps_pi3_se3_90_900px',
    'Pi3-long/all_poses/exps_pi3_se3_120_900px',

    'Pi3-long/all_poses/exps_pi3x_se3_30',
    'Pi3-long/all_poses/exps_pi3x_se3_60',
    'Pi3-long/all_poses/exps_pi3x_se3_90',
    'Pi3-long/all_poses/exps_pi3x_se3_120',
    
    'Pi3-long/all_poses/exps_pi3x_se3_30_900px',
    'Pi3-long/all_poses/exps_pi3x_se3_60_900px',
    'Pi3-long/all_poses/exps_pi3x_se3_90_900px',
    'Pi3-long/all_poses/exps_pi3x_se3_120_900px',

    'Pi3-long/all_poses/exps_pi3_sim3_30',
    'Pi3-long/all_poses/exps_pi3_sim3_60',
    'Pi3-long/all_poses/exps_pi3_sim3_90',
    'Pi3-long/all_poses/exps_pi3_sim3_120',
    'Pi3-long/all_poses/exps_pi3_sim3_30_900px',
    'Pi3-long/all_poses/exps_pi3_sim3_60_900px',
    'Pi3-long/all_poses/exps_pi3_sim3_90_900px',
    'Pi3-long/all_poses/exps_pi3_sim3_120_900px',
    'Pi3-long/all_poses/exps_pi3x_sim3_30',
    'Pi3-long/all_poses/exps_pi3x_sim3_60',
    'Pi3-long/all_poses/exps_pi3x_sim3_90',
    'Pi3-long/all_poses/exps_pi3x_sim3_120',
    'Pi3-long/all_poses/exps_pi3x_sim3_30_900px',
    'Pi3-long/all_poses/exps_pi3x_sim3_60_900px',
    'Pi3-long/all_poses/exps_pi3x_sim3_90_900px',
    'Pi3-long/all_poses/exps_pi3x_sim3_120_900px',
]

# GT数据根目录 (包含00.txt, 01.txt, ..., 10.txt等文件)
GT_ROOT = './ATE-Eval/KITTI/KITTI-GT/dataset/poses'

# 要处理的序列号列表 (默认00-10)
SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# 是否跳过缺失的序列
SKIP_MISSING = True

# 是否处理所有时间戳实验 (True: 处理所有并取平均值, False: 只处理最新的一个)
PROCESS_ALL_TIMESTAMPS = False

# 汇总Excel文件输出路径（None则自动保存在当前目录）
OUTPUT_EXCEL = './ATE_summary.xlsx'  # 例如 './ATE_summary.xlsx'

# ==================== 以下为功能代码 ====================

def load_poses_kitti(pose_file, index=False, inv=False):
    """加载KITTI格式的位姿文件"""
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
    """计算轨迹长度"""
    translations = poses[:, :3, 3]
    distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    total_length = np.sum(distances)
    return total_length

def find_camera_poses_files(exp_dir):
    """在实验目录中查找所有camera_poses.txt文件"""
    camera_poses_files = []
    
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file == "camera_poses.txt":
                file_path = os.path.join(root, file)
                timestamp_dir = os.path.basename(root)
                if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', timestamp_dir):
                    mtime = os.path.getmtime(file_path)
                    camera_poses_files.append({
                        'path': file_path,
                        'timestamp': timestamp_dir,
                        'mtime': mtime,
                        'full_path': root
                    })
    
    camera_poses_files.sort(key=lambda x: x['mtime'], reverse=True)
    return camera_poses_files

def find_exp_directory(exp_root, seq):
    """根据实验根目录和序列号查找实验子目录"""
    for item in os.listdir(exp_root):
        item_path = os.path.join(exp_root, item)
        if os.path.isdir(item_path):
            if f"_{seq}_" in item or item.endswith(f"_{seq}"):
                return item_path
    
    patterns = [
        f"*{seq}*",
        f"*sequences_{seq}_*",
        f"*{seq}_image_*",
    ]
    
    for pattern in patterns:
        import glob
        matches = glob.glob(os.path.join(exp_root, pattern))
        for match in matches:
            if os.path.isdir(match):
                return match
    
    return None

def process_single_experiment(exp_info, seq, gt_file):
    """处理单个时间戳实验，返回ATE RMSE等指标"""
    camera_poses_file = exp_info['path']
    timestamp = exp_info['timestamp']
    exp_dir = exp_info['full_path']
    
    print(f"\n  - 处理实验: {timestamp}")
    print(f"    实验文件: {camera_poses_file}")
    print(f"    GT文件: {gt_file}")
    
    try:
        pose_est_data = load_poses_kitti(camera_poses_file)
        pose_gt_data = load_poses_kitti(gt_file)
    except Exception as e:
        print(f"    加载数据失败: {e}")
        return None
    
    trajectory_length = calculate_trajectory_length(pose_gt_data)
    print(f"    轨迹长度: {trajectory_length:.2f} 米")
    
    timestamps_gt = np.linspace(0, 1, pose_gt_data.shape[0])
    timestamps_est = np.linspace(0, 1, pose_est_data.shape[0])
    
    num_samples = pose_est_data.shape[0]
    indices = np.linspace(0, pose_gt_data.shape[0] - 1, num_samples, dtype=int)
    pose_gt_data_sampled = pose_gt_data[indices]
    timestamps_gt_sampled = timestamps_gt[indices]
    
    traj_est = et.PoseTrajectory3D(poses_se3=pose_est_data, timestamps=timestamps_est)
    traj_gt = et.PoseTrajectory3D(poses_se3=pose_gt_data_sampled, timestamps=timestamps_gt_sampled)
    
    traj_ref, traj_est_aligned = sync.associate_trajectories(traj_gt, traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True)
    
    ape_metric = APE(PoseRelation.full_transformation)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ape_stats = ape_metric.get_all_statistics()
    
    print(f"    ATE RMSE: {ape_stats['rmse']:.4f} 米")
    
    rpe_metric = RPE(PoseRelation.full_transformation)
    rpe_metric.process_data((traj_ref, traj_est_aligned))
    rpe_stats = rpe_metric.get_all_statistics()
    
    timestamp_str = timestamp.replace('-', '')
    results_file = os.path.join(exp_dir, f'ATE_{timestamp_str}.txt')
    with open(results_file, 'w') as f:
        f.write(f"序列: {seq}\n")
        f.write(f"实验时间戳: {timestamp}\n")
        f.write(f"ATE RMSE: {ape_stats['rmse']:.4f} 米\n")
        f.write(f"ATE 均值: {ape_stats['mean']:.4f} 米\n")
        f.write(f"ATE 中位数: {ape_stats['median']:.4f} 米\n")
        f.write(f"ATE 标准差: {ape_stats['std']:.4f} 米\n")
        f.write(f"ATE 最小值: {ape_stats['min']:.4f} 米\n")
        f.write(f"ATE 最大值: {ape_stats['max']:.4f} 米\n")
        f.write(f"RPE RMSE: {rpe_stats['rmse']:.4f} 米\n")
        f.write(f"轨迹长度: {trajectory_length:.2f} 米\n")
        f.write(f"使用的估计文件: {camera_poses_file}\n")
        f.write(f"使用的GT文件: {gt_file}\n")
    
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_aspect('equal', 'box')
    ax.set_box_aspect(1)
    color_dark_red = '#B6443F'
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'GT Poses', linewidth=4.5)
    plot.traj(ax, plot_mode, traj_est_aligned, '-', color_dark_red, 'Estimated Poses', linewidth=4.5)
    
    for ext in ['png', 'pdf']:
        pic_path = os.path.join(exp_dir, f"trajectory_plot_{timestamp_str}.{ext}")
        fig.savefig(pic_path, dpi=300)
        print(f"    轨迹图已保存为: {pic_path}")
    plt.close(fig)
    
    return {
        'seq': seq,
        'timestamp': timestamp,
        'ate_rmse': ape_stats['rmse'],
        'rpe_rmse': rpe_stats['rmse'],
        'trajectory_length': trajectory_length,
        'num_poses': pose_est_data.shape[0],
        'results_file': results_file,
        'exp_dir': exp_dir,
        'experiment_path': camera_poses_file
    }

def process_sequence(exp_root, seq, gt_root, skip_missing, process_all):
    """处理单个序列下的所有时间戳实验，返回结果列表"""
    print(f"\n{'='*60}")
    print(f"处理序列: {seq}")
    print(f"{'='*60}")
    
    exp_dir = find_exp_directory(exp_root, seq)
    if exp_dir is None:
        print(f"未找到序列 {seq} 的实验目录")
        return []
    
    print(f"实验目录: {exp_dir}")
    
    gt_file = os.path.join(gt_root, f"{seq}.txt")
    if not os.path.exists(gt_file):
        print(f"GT文件不存在: {gt_file}")
        return []
    
    exp_files = find_camera_poses_files(exp_dir)
    if not exp_files:
        print(f"未找到任何camera_poses.txt文件: {exp_dir}")
        return []
    
    print(f"找到 {len(exp_files)} 个时间戳实验:")
    for i, exp in enumerate(exp_files, 1):
        print(f"  {i}. {exp['timestamp']} ({datetime.fromtimestamp(exp['mtime']).strftime('%Y-%m-%d %H:%M:%S')})")
    
    if process_all:
        experiments_to_process = exp_files
        print(f"\n处理所有 {len(exp_files)} 个实验...")
    else:
        experiments_to_process = [exp_files[0]]
        print(f"\n只处理最新的实验: {exp_files[0]['timestamp']}")
    
    all_results = []
    for exp_info in experiments_to_process:
        result = process_single_experiment(exp_info, seq, gt_file)
        if result:
            all_results.append(result)
    
    return all_results

def evaluate_single_exp_root(exp_root, gt_root, sequences, skip_missing, process_all):
    """
    评估单个实验根目录，返回每个序列的平均ATE RMSE（基于process_all决定是平均所有实验还是只取最新）
    """
    print(f"\n{'#'*60}")
    print(f"开始评估实验根目录: {exp_root}")
    print(f"{'#'*60}")
    
    if not os.path.exists(exp_root):
        print(f"错误: 实验根目录不存在: {exp_root}")
        return {}
    
    all_results = []  # 存储所有实验结果（每个实验一行）
    
    for seq in sequences:
        try:
            seq_results = process_sequence(exp_root, seq, gt_root, skip_missing, process_all)
            all_results.extend(seq_results)
        except Exception as e:
            if skip_missing:
                print(f"处理序列 {seq} 时出错: {e}")
                print(f"跳过序列 {seq}")
            else:
                print(f"处理序列 {seq} 时出错: {e}")
                raise
    
    # 按序列分组，计算平均ATE RMSE
    seq_ate_mean = {}
    for seq in sequences:
        seq_ates = [r['ate_rmse'] for r in all_results if r['seq'] == seq]
        if seq_ates:
            seq_ate_mean[seq] = np.mean(seq_ates)
        else:
            seq_ate_mean[seq] = np.nan  # 无数据时用NaN填充
    
    # 可选：打印该实验根目录的汇总
    print(f"\n--- 实验根目录 {exp_root} 各序列平均ATE RMSE (米) ---")
    for seq in sequences:
        val = seq_ate_mean[seq]
        if np.isnan(val):
            print(f"  序列 {seq}: 无有效数据")
        else:
            print(f"  序列 {seq}: {val:.4f}")
    
    return seq_ate_mean

def main():
    """主函数：处理多个EXP_ROOT，生成汇总Excel"""
    print("="*60)
    print("自动化ATE/RPE评估脚本 - 支持多个实验根目录")
    print(f"实验根目录列表: {EXP_ROOTS}")
    print(f"GT根目录: {GT_ROOT}")
    print(f"要处理的序列: {SEQUENCES}")
    print(f"处理所有时间戳实验并取平均: {PROCESS_ALL_TIMESTAMPS}")
    print("="*60)
    
    if not os.path.exists(GT_ROOT):
        print(f"错误: GT根目录不存在: {GT_ROOT}")
        return
    
    if not EXP_ROOTS:
        print("错误: EXP_ROOTS列表为空，请至少添加一个实验根目录")
        return
    
    # 存储每个exp_root的序列平均ATE
    data_rows = []  # 每个元素是一个字典 {exp_root_name, seq_00, seq_01, ..., seq_10}
    
    for exp_root in EXP_ROOTS:
        exp_name = os.path.basename(exp_root)  # 用于Excel行标签
        seq_avg_ate = evaluate_single_exp_root(
            exp_root=exp_root,
            gt_root=GT_ROOT,
            sequences=SEQUENCES,
            skip_missing=SKIP_MISSING,
            process_all=PROCESS_ALL_TIMESTAMPS
        )
        # 构建行数据
        row = {'EXP_ROOT': exp_name}
        row.update({seq: seq_avg_ate.get(seq, np.nan) for seq in SEQUENCES})
        data_rows.append(row)
    
    # 构建DataFrame
    df = pd.DataFrame(data_rows)
    # 计算AVG和AVG (w/o 01)
    seq_cols = SEQUENCES  # 列表形式 ['00','01',...,'10']
    # 计算每行的AVG（忽略NaN）
    df['AVG'] = df[seq_cols].mean(axis=1, skipna=True)
    # 计算排除序列01后的平均值
    seq_cols_without_01 = [s for s in seq_cols if s != '01']
    df['AVG (w/o 01)'] = df[seq_cols_without_01].mean(axis=1, skipna=True)
    
    # 调整列顺序：EXP_ROOT, AVG, AVG (w/o 01), 然后00-10
    ordered_cols = ['EXP_ROOT', 'AVG', 'AVG (w/o 01)'] + seq_cols
    df = df[ordered_cols]
    
    # 输出Excel
    if OUTPUT_EXCEL is None:
        output_path = os.path.join(os.getcwd(), 'ATE_summary_multiple_exp.xlsx')
    else:
        output_path = OUTPUT_EXCEL
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        df.to_excel(output_path, index=False, float_format='%.4f')
        print(f"\n汇总Excel表格已保存至: {output_path}")
    except Exception as e:
        print(f"保存Excel失败: {e}")
        print("请确保已安装 pandas 和 openpyxl (pip install pandas openpyxl)")
    
    # 同时打印表格到控制台
    print("\n" + "="*80)
    print("汇总结果（ATE RMSE，单位：米）:")
    print(df.to_string(index=False, float_format='%.4f', na_rep='NaN'))
    print("="*80)
    
    print(f"\n处理完成！")

if __name__ == '__main__':
    main()