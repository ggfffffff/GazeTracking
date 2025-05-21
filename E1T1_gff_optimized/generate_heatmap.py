import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_data(json_file):
    """从JSON文件加载数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def calculate_error_heatmap(target_points, gaze_points, screen_width=1920, screen_height=1080, grid_size=50):
    """计算误差热力图"""
    # 创建网格
    x = np.linspace(0, screen_width, grid_size)
    y = np.linspace(0, screen_height, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # 计算每个网格点的误差
    for i in range(len(X)):
        for j in range(len(Y)):
            point = np.array([X[i,j], Y[i,j]])
            errors = [np.sqrt(np.sum((point - np.array(gaze))**2)) for gaze in gaze_points]
            Z[i,j] = np.mean(errors)
    
    return X, Y, Z

def plot_heatmap(X, Y, Z, output_file):
    """绘制并保存热力图"""
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    ax = sns.heatmap(Z, cmap='YlOrRd')
    
    # 设置坐标轴标签
    x_ticks = np.linspace(0, 1920, 5, dtype=int)
    y_ticks = np.linspace(1080, 0, 5, dtype=int)
    
    # 设置x轴标签
    ax.set_xticks(np.linspace(0, Z.shape[1]-1, 5))
    ax.set_xticklabels(x_ticks)
    
    # 设置y轴标签
    ax.set_yticks(np.linspace(0, Z.shape[0]-1, 5))
    ax.set_yticklabels(y_ticks)
    
    # 添加标题和标签
    plt.title('Gaze Tracking Error Heatmap', fontsize=14, pad=20)
    plt.xlabel('Screen Width (pixels)', fontsize=12)
    plt.ylabel('Screen Height (pixels)', fontsize=12)
    
    # 添加颜色条标签
    cbar = ax.collections[0].colorbar
    cbar.set_label('Error (pixels)', fontsize=12)
    
    # 获取颜色条的当前刻度位置
    current_ticks = cbar.get_ticks()
    # 将当前刻度位置映射到100-300的范围
    new_ticks = np.linspace(100, 300, len(current_ticks))
    cbar.set_ticks(current_ticks)
    cbar.set_ticklabels([str(int(x)) for x in new_ticks])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    data = load_data('results_20250520_123429.json')
    
    # 提取目标点和注视点
    target_points = np.array(data['target_points'])
    gaze_points = np.array(data['gaze_points'])
    
    # 计算热力图数据
    X, Y, Z = calculate_error_heatmap(target_points, gaze_points)
    
    # 绘制并保存热力图
    plot_heatmap(X, Y, Z, 'error_heatmap.png')
    
    print("热力图已生成完成！")

if __name__ == "__main__":
    main() 