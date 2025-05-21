import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(json_file):
    """从JSON文件加载数据"""
    with open(json_file, 'r') as f:
        return json.load(f)

def prepare_boxplot_data(data):
    """准备箱图数据"""
    models = ['basic', 'coordinate', 'kalman', 'multifilter', 'optimized', 'beam_eye_tracker']
    mae_data = []
    error_95_data = []
    labels = []
    
    for model in models:
        model_data = data[model]['subjects']
        # 收集MAE数据
        mae_values = [subject['MAE'] for subject in model_data.values()]
        mae_data.append(mae_values)
        # 收集95%误差数据
        error_95_values = [subject['95th_percentile_error'] for subject in model_data.values()]
        error_95_data.append(error_95_values)
        # 添加标签
        labels.append(model.replace('_', ' ').title())
    
    return mae_data, error_95_data, labels

def plot_boxplots(mae_data, error_95_data, labels):
    """绘制箱图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制MAE箱图
    sns.boxplot(data=mae_data, ax=ax1, showfliers=True, palette='Set3', whis=1.5)  # 使用柔和的颜色
    # 添加散点显示所有数据
    for i in range(len(mae_data)):
        x = np.random.normal(i, 0.04, size=len(mae_data[i]))
        ax1.scatter(x, mae_data[i], alpha=0.5, color='gray', s=20)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title('MAE分布对比', fontsize=12, pad=20)
    ax1.set_ylabel('MAE (像素)', fontsize=10)
    
    # 绘制95%误差箱图
    sns.boxplot(data=error_95_data, ax=ax2, showfliers=True, palette='Set3', whis=1.5)  # 使用柔和的颜色
    # 添加散点显示所有数据
    for i in range(len(error_95_data)):
        x = np.random.normal(i, 0.04, size=len(error_95_data[i]))
        ax2.scatter(x, error_95_data[i], alpha=0.5, color='gray', s=20)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_title('95%误差分布对比', fontsize=12, pad=20)
    ax2.set_ylabel('95%误差 (像素)', fontsize=10)
    
    # 添加图例说明
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.5, label='数据分布范围'),
        plt.Line2D([0], [0], color='black', label='中位数'),
        plt.Line2D([0], [0], marker='o', color='gray', label='实际数据点',
                  markerfacecolor='gray', markersize=5)
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=3, frameon=False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('model_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    data = load_data('all_subjects_results.json')
    
    # 准备箱图数据
    mae_data, error_95_data, labels = prepare_boxplot_data(data)
    
    # 绘制箱图
    plot_boxplots(mae_data, error_95_data, labels)
    
    print("箱图已生成完成！")

if __name__ == "__main__":
    main() 