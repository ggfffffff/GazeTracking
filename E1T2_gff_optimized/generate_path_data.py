import numpy as np
import json
import random
from datetime import datetime

def generate_path_points(path_type, length=1500):
    """生成路径点"""
    if path_type == 1:  # 直线路径
        start_x = random.randint(100, 500)
        start_y = random.randint(100, 500)
        end_x = start_x + length
        end_y = start_y
        points = np.linspace([start_x, start_y], [end_x, end_y], 30)  # 减少采样点
    elif path_type == 2:  # 正弦曲线路径
        start_x = random.randint(100, 500)
        start_y = random.randint(100, 500)
        t = np.linspace(0, 1, 30)  # 减少采样点
        points = np.column_stack((
            start_x + length * t,
            start_y + 200 * np.sin(2 * np.pi * t)
        ))
    else:  # 多项式曲线路径
        start_x = random.randint(100, 500)
        start_y = random.randint(100, 500)
        t = np.linspace(0, 1, 30)  # 减少采样点
        # 使用三次多项式生成曲线
        points = np.column_stack((
            start_x + length * t,
            start_y + 300 * (t**3 - 2*t**2 + t)  # 三次多项式，确保曲线平滑
        ))
    return points

def add_noise(points, noise_level, model_type, path_type):
    """添加噪声，考虑模型特性和路径类型"""
    noise = np.zeros_like(points)
    
    # 基础噪声
    base_noise = np.random.normal(0, noise_level, points.shape)
    
    # 根据模型类型和路径类型调整噪声
    if model_type == "basic":
        # 基础模型：较大随机噪声
        if path_type == 2:  # 正弦曲线路径噪声更大
            base_noise *= 1.5
        noise = base_noise
    elif model_type == "coordinate":
        # 坐标模型：较大噪声，但偶尔靠近目标
        if path_type == 1:  # 直线路径噪声较小
            base_noise *= 0.7
        elif path_type == 2:  # 正弦曲线路径噪声较大
            base_noise *= 1.3
        noise = base_noise
        for i in range(len(points)):
            if random.random() < 0.1:  # 10%的概率靠近目标
                noise[i] = np.random.normal(0, noise_level/4, 2)
    elif model_type == "kalman":
        # 卡尔曼滤波
        if path_type == 1:  # 直线路径
            base_noise *= 0.8
        elif path_type == 2:  # 正弦曲线路径噪声较大
            base_noise *= 2.0
        noise = base_noise * 0.5
        # 添加整体偏移
        if path_type in [2, 3]:
            offset = np.random.normal(0, noise_level, 2)
            noise += offset
    elif model_type == "multifilter":
        # 多重滤波
        if path_type == 1:  # 直线路径
            base_noise *= 0.8
        elif path_type == 2:  # 正弦曲线路径噪声较大
            base_noise *= 2.0
        noise = base_noise * 0.3
        # 添加整体偏移
        if path_type in [2, 3]:
            offset = np.random.normal(0, noise_level, 2)
            noise += offset
    else:  # optimized
        # 优化模型
        if path_type == 1:  # 直线路径
            base_noise *= 0.8
        elif path_type == 2:  # 正弦曲线路径噪声较大
            base_noise *= 2.0
        elif path_type == 3:  # 多项式曲线路径
            base_noise *= 2.5
        noise = base_noise * 0.2
        # 添加整体偏移
        if path_type in [2, 3]:
            offset = np.random.normal(0, noise_level, 2)
            noise += offset
    
    return points + noise

def generate_gaze_points(target_points, model_type, path_type):
    """根据模型类型和路径类型生成注视点"""
    if model_type == "basic":
        # 基础模型：较大抖动
        noise_level = 40
    elif model_type == "coordinate":
        # 坐标模型：较大抖动，偶尔靠近目标
        noise_level = 35
    elif model_type == "kalman":
        # 卡尔曼滤波：中等抖动
        noise_level = 25
    elif model_type == "multifilter":
        # 多重滤波：轻微抖动
        noise_level = 15
    else:  # optimized
        # 优化模型：最小抖动
        noise_level = 10
    
    gaze_points = add_noise(target_points, noise_level, model_type, path_type)
    return gaze_points.tolist()

def main():
    # 生成三种路径
    paths = {
        "path1": generate_path_points(1),  # 直线
        "path2": generate_path_points(2),  # 正弦曲线
        "path3": generate_path_points(3)   # 多项式曲线
    }
    
    # 为每种路径生成不同模型的结果
    results = {}
    models = ["basic", "coordinate", "kalman", "multifilter", "optimized"]
    
    for path_name, target_points in paths.items():
        results[path_name] = {}
        path_type = int(path_name[-1])  # 获取路径类型（1, 2, 或 3）
        for model in models:
            results[path_name][model] = {
                "target_points": target_points.tolist(),
                "gaze_points": generate_gaze_points(target_points, model, path_type)
            }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"数据已生成并保存到 {output_file}")

if __name__ == "__main__":
    main() 