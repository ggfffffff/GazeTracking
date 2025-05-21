import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(json_file):
    """加载JSON数据"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_trajectories(data):
    """绘制轨迹对比图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建3x5的子图
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('不同模型在不同路径下的轨迹对比', fontsize=16, y=0.95)
    
    # 设置颜色
    target_color = '#2ecc71'  # 目标轨迹颜色
    gaze_colors = {
        'basic': '#e74c3c',      # 红色
        'coordinate': '#e67e22',  # 橙色
        'kalman': '#3498db',     # 蓝色
        'multifilter': '#9b59b6', # 紫色
        'optimized': '#1abc9c'    # 青色
    }
    
    # 遍历每个路径和模型
    for i, path_name in enumerate(['path1', 'path2', 'path3']):
        for j, model in enumerate(['basic', 'coordinate', 'kalman', 'multifilter', 'optimized']):
            ax = axes[i, j]
            
            # 获取数据
            target_points = np.array(data[path_name][model]['target_points'])
            gaze_points = np.array(data[path_name][model]['gaze_points'])
            
            # 绘制目标轨迹
            ax.plot(target_points[:, 0], target_points[:, 1], 
                   color=target_color, label='目标轨迹', linewidth=2, alpha=0.7)
            
            # 绘制注视轨迹
            ax.plot(gaze_points[:, 0], gaze_points[:, 1], 
                   color=gaze_colors[model], label='注视轨迹', linewidth=1.5, alpha=0.6)
            
            # 设置标题和标签
            if i == 0:
                ax.set_title(f'{model.capitalize()}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'Path {i+1}', fontsize=12)
            
            # 设置坐标轴范围
            ax.set_xlim(min(target_points[:, 0])-100, max(target_points[:, 0])+100)
            ax.set_ylim(min(target_points[:, 1])-100, max(target_points[:, 1])+100)
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 只在第一个子图添加图例
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    data = load_data('results_20250521_143505.json')
    
    # 绘制轨迹对比图
    plot_trajectories(data)
    print("轨迹对比图已生成完成！")

if __name__ == "__main__":
    main() 