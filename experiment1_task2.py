import cv2
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
import os
from gaze_tracking import GazeTracking
from filterpy.kalman import KalmanFilter

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TrackingStabilityTask:
    def __init__(self, subject_name, model_type):
        self.screen_width = 1920
        self.screen_height = 1080
        self.target_radius = 10  # 目标点半径
        self.gaze_threshold = 100  # 注视判定阈值（像素）
        self.num_paths = 3  # 路径数量
        
        # 存储实验数据
        self.paths = []  # 路径数据
        self.gaze_points = []  # 注视点数据
        self.filtered_points = []  # 滤波后的注视点
        self.raw_gaze_points = []  # 原始注视点数据
        
        # 创建结果目录
        self.results_dir = f"E1T2_{subject_name}_{model_type}"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # 眼动追踪相关参数
        self.calibration_file = "calibration-5.json"
        self.moving_average_window = 5
        self.dead_zone = 5
        self.min_velocity_threshold = 0.5
        self.y_offset = -10
        
        # 初始化眼动追踪
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.calibration_data = self.load_calibration()
        
        # 初始化卡尔曼滤波器
        self.kf = self.create_kalman_filter_acc()
        self.initialized = False
        self.last_filtered = None
        self.recent_points = []
        self.last_stable_point = None
        self.last_update_time = time.time()
        
        # 生成路径
        self.generate_paths()
        
    def load_calibration(self):
        """加载校准数据"""
        try:
            with open(self.calibration_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("校准文件未找到，请先运行校准程序。")
            return None
            
    def get_gaze_coordinates(self, hr, vr):
        """根据校准数据，将gaze tracking的水平&垂直比例转换为屏幕坐标"""
        if not self.calibration_data:
            return None, None

        x_vals = np.array([p["hr"] for p in self.calibration_data])
        y_vals = np.array([p["vr"] for p in self.calibration_data])
        screen_x_vals = np.array([p["x"] for p in self.calibration_data])
        screen_y_vals = np.array([p["y"] for p in self.calibration_data])

        poly_x = np.polyfit(x_vals, screen_x_vals, 2)
        poly_y = np.polyfit(y_vals, screen_y_vals, 2)

        gaze_x = int(np.polyval(poly_x, hr))
        gaze_y = int(np.polyval(poly_y, vr)) + self.y_offset
        return gaze_x, gaze_y
        
    def create_kalman_filter_acc(self):
        """创建基于常加速模型的卡尔曼滤波器"""
        kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt,       0],
            [0, 0, 0, 1, 0,       dt],
            [0, 0, 0, 0, 1,        0],
            [0, 0, 0, 0, 0,        1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        kf.P *= 10.
        kf.R *= 20.
        kf.Q *= 0.01
        return kf
        
    def low_pass_filter(self, new_point, last_filtered, alpha_x=0.2, alpha_y=0.3):
        """对新坐标进行指数加权低通滤波"""
        if last_filtered is None:
            return new_point
        filtered_x = int(alpha_x * new_point[0] + (1 - alpha_x) * last_filtered[0])
        filtered_y = int(alpha_y * new_point[1] + (1 - alpha_y) * last_filtered[1])
        return (filtered_x, filtered_y)
        
    def moving_average(self, points):
        """计算移动平均点"""
        if len(points) < 2:
            return points[-1]
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (int(x_sum / len(points)), int(y_sum / len(points)))
        
    def is_in_dead_zone(self, point1, point2):
        """判断两点是否在死区内"""
        return abs(point1[0] - point2[0]) <= self.dead_zone and abs(point1[1] - point2[1]) <= self.dead_zone
        
    def get_filtered_gaze_point(self):
        """获取经过滤波处理的注视点坐标"""
        ret, frame = self.webcam.read()
        if not ret:
            return None
            
        self.gaze.refresh(frame)
        hr = self.gaze.horizontal_ratio()
        vr = self.gaze.vertical_ratio()
        
        if hr is not None and vr is not None:
            gaze_x, gaze_y = self.get_gaze_coordinates(hr, vr)
            if gaze_x is not None and gaze_y is not None:
                if not self.initialized:
                    self.kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    self.last_filtered = (gaze_x, gaze_y)
                    self.initialized = True
                    
                self.kf.predict()
                self.kf.update([gaze_x, gaze_y])
                kalman_point = (int(self.kf.x[0]), int(self.kf.x[1]))
                
                current_velocity = np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)
                
                filtered_point = self.low_pass_filter(kalman_point, self.last_filtered,
                                                    alpha_x=0.2, alpha_y=0.1)
                
                self.recent_points.append(filtered_point)
                if len(self.recent_points) > self.moving_average_window:
                    self.recent_points.pop(0)
                averaged_point = self.moving_average(self.recent_points)
                
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                if current_velocity < self.min_velocity_threshold:
                    if self.last_stable_point is not None:
                        averaged_point = self.last_stable_point
                elif self.is_in_dead_zone(averaged_point, self.last_filtered):
                    averaged_point = self.last_filtered
                
                self.last_filtered = averaged_point
                self.last_stable_point = averaged_point
                self.last_update_time = current_time
                
                return averaged_point
        return None
        
    def generate_paths(self):
        """生成三条不同形状的路径"""
        # 定义屏幕中心区域
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        path_length = 400  # 减小路径长度，确保在中心区域
        
        # 1. 水平直线路径
        start_x1 = center_x - path_length // 2
        start_y1 = center_y
        end_x1 = center_x + path_length // 2
        end_y1 = center_y
        self.paths.append({
            'start': (start_x1, start_y1),
            'end': (end_x1, end_y1),
            'duration': 4.0,  # 4秒完成
            'type': 'linear'
        })
        
        # 2. 45度斜线路径
        start_x2 = center_x - path_length // 2
        start_y2 = center_y + path_length // 2
        end_x2 = center_x + path_length // 2
        end_y2 = center_y - path_length // 2
        self.paths.append({
            'start': (start_x2, start_y2),
            'end': (end_x2, end_y2),
            'duration': 4.0,
            'type': 'linear'
        })
        
        # 3. 平滑曲线路径（使用贝塞尔曲线）
        # 控制点
        p0 = (center_x - path_length // 2, center_y)  # 起点
        p1 = (center_x - path_length // 4, center_y - path_length // 2)  # 控制点1
        p2 = (center_x + path_length // 4, center_y + path_length // 2)  # 控制点2
        p3 = (center_x + path_length // 2, center_y)  # 终点
        
        self.paths.append({
            'start': p0,
            'end': p3,
            'control_points': [p1, p2],
            'duration': 4.0,
            'type': 'bezier'
        })
        
    def get_bezier_point(self, p0, p1, p2, p3, t):
        """计算贝塞尔曲线上的点"""
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        return (int(x), int(y))
        
    def calculate_tde(self, target_points, gaze_points):
        """计算轨迹偏移误差（Trajectory Deviation Error）"""
        errors = []
        for target, gaze in zip(target_points, gaze_points):
            error = np.sqrt((target[0] - gaze[0])**2 + (target[1] - gaze[1])**2)
            errors.append(error)
        return np.mean(errors)
        
    def calculate_smoothness(self, points):
        """计算平滑度（相邻点之间速度变化程度）"""
        if len(points) < 3:
            return 0
            
        velocities = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
            
        # 计算速度变化的标准差
        velocity_changes = np.diff(velocities)
        return np.std(velocity_changes)
        
    def run_experiment(self):
        """运行实验"""
        if not self.calibration_data:
            print("未找到校准数据，请先运行校准程序。")
            return
            
        # 创建全屏窗口
        cv2.namedWindow("Experiment", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for path_idx, path in enumerate(self.paths):
            print(f"开始第 {path_idx + 1} 条路径...")
            
            # 当前路径的数据
            current_path_points = []
            current_gaze_points = []
            current_raw_gaze_points = []
            
            # 准备阶段：显示固定起始点，等待用户视线稳定
            print("请将视线移动到绿色起始点...")
            stable_count = 0
            required_stable_frames = 30
            start_time = time.time()
            
            while stable_count < required_stable_frames:
                frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                cv2.circle(frame, path['start'], self.target_radius, (0, 255, 0), -1)
                
                gaze_point = self.get_filtered_gaze_point()
                if gaze_point is not None:
                    cv2.circle(frame, gaze_point, 5, (0, 0, 255), -1)
                    
                    distance = np.sqrt((gaze_point[0] - path['start'][0])**2 + (gaze_point[1] - path['start'][1])**2)
                    if distance < self.gaze_threshold:
                        stable_count += 1
                    else:
                        stable_count = 0
                
                cv2.imshow("Experiment", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    return
                    
                if time.time() - start_time > 5:
                    break
            
            print("开始追踪...")
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < path['duration']:
                t = (time.time() - start_time) / path['duration']
                
                if path['type'] == 'linear':
                    # 线性插值
                    current_x = int(path['start'][0] + t * (path['end'][0] - path['start'][0]))
                    current_y = int(path['start'][1] + t * (path['end'][1] - path['start'][1]))
                    current_point = (current_x, current_y)
                else:
                    # 贝塞尔曲线
                    current_point = self.get_bezier_point(
                        path['start'],
                        path['control_points'][0],
                        path['control_points'][1],
                        path['end'],
                        t
                    )
                
                current_path_points.append(current_point)
                
                gaze_point = self.get_filtered_gaze_point()
                if gaze_point is not None:
                    current_gaze_points.append(gaze_point)
                    current_raw_gaze_points.append(gaze_point)
                
                frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                cv2.circle(frame, current_point, self.target_radius, (0, 255, 0), -1)
                
                if gaze_point is not None:
                    cv2.circle(frame, gaze_point, 5, (0, 0, 255), -1)
                
                cv2.imshow("Experiment", frame)
                
                frame_count += 1
                time.sleep(max(0, 1/30 - (time.time() - start_time - frame_count/30)))
                
                if cv2.waitKey(1) & 0xFF == 27:
                    return
                    
            self.paths[path_idx]['points'] = current_path_points
            self.gaze_points.append(current_gaze_points)
            self.raw_gaze_points.append(current_raw_gaze_points)
            
            time.sleep(1)
            
        cv2.destroyAllWindows()
        self.webcam.release()
        
    def calculate_metrics(self):
        """计算评价指标"""
        metrics = []
        for path_idx, path in enumerate(self.paths):
            target_points = path['points']
            gaze_points = self.gaze_points[path_idx]
            
            # 计算TDE
            tde = self.calculate_tde(target_points, gaze_points)
            
            # 计算平滑度
            smoothness = self.calculate_smoothness(gaze_points)
            
            metrics.append({
                'path_index': path_idx + 1,
                'TDE': tde,
                'smoothness': smoothness
            })
            
        return metrics
        
    def generate_visualization(self):
        """生成轨迹可视化图"""
        plt.figure(figsize=(15, 5))
        
        for path_idx, path in enumerate(self.paths):
            plt.subplot(1, 3, path_idx + 1)
            
            # 绘制理想轨迹
            target_points = path['points']
            target_x = [p[0] for p in target_points]
            target_y = [p[1] for p in target_points]
            plt.plot(target_x, target_y, 'g-', label='Ideal Trajectory', alpha=0.5)
            
            # 绘制原始注视轨迹
            raw_gaze_points = self.raw_gaze_points[path_idx]
            raw_x = [p[0] for p in raw_gaze_points]
            raw_y = [p[1] for p in raw_gaze_points]
            plt.plot(raw_x, raw_y, 'r.', label='Raw Gaze Points', alpha=0.3)
            
            # 绘制滤波后的轨迹
            gaze_points = self.gaze_points[path_idx]
            gaze_x = [p[0] for p in gaze_points]
            gaze_y = [p[1] for p in gaze_points]
            plt.plot(gaze_x, gaze_y, 'b-', label='Filtered Trajectory')
            
            plt.title(f'Path {path_idx + 1}')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'trajectory_comparison.png'))
        plt.close()
        
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "metrics": self.calculate_metrics(),
            "paths": self.paths,
            "gaze_points": self.gaze_points,
            "raw_gaze_points": self.raw_gaze_points
        }
        
        # 保存JSON格式的完整数据
        with open(os.path.join(self.results_dir, f'results_{timestamp}.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # 保存CSV格式的原始数据
        for path_idx, path in enumerate(self.paths):
            # 获取目标点和注视点
            target_points = path['points']
            gaze_points = self.gaze_points[path_idx]
            
            # 确保数据长度一致
            min_length = min(len(target_points), len(gaze_points))
            target_points = target_points[:min_length]
            gaze_points = gaze_points[:min_length]
            
            # 准备数据
            raw_data = np.column_stack((
                [p[0] for p in target_points],
                [p[1] for p in target_points],
                [p[0] for p in gaze_points],
                [p[1] for p in gaze_points]
            ))
            
            # 保存数据
            np.savetxt(
                os.path.join(self.results_dir, f'raw_data_path{path_idx+1}_{timestamp}.csv'),
                raw_data,
                delimiter=',',
                header='target_x,target_y,gaze_x,gaze_y',
                comments=''
            )
            
    def run(self):
        """运行完整的实验流程"""
        print("开始实验...")
        self.run_experiment()
        print("计算评价指标...")
        metrics = self.calculate_metrics()
        for m in metrics:
            print(f"路径 {m['path_index']}:")
            print(f"  TDE: {m['TDE']:.2f}像素")
            print(f"  平滑度: {m['smoothness']:.2f}")
        print("生成可视化...")
        self.generate_visualization()
        print("保存结果...")
        self.save_results()
        print("实验完成！")

if __name__ == "__main__":
    # 在这里修改受试者姓名和模型类型
    subject_name = "gff"  # 修改为实际的受试者姓名
    
    # 可用的模型类型：
    # "basic" - 基础版本
    # "coordinate" - 坐标映射版本
    # "kalman" - 卡尔曼滤波版本
    # "multifilter" - 多重滤波版本
    # "optimized" - 优化版本
    model_type = "optimized"  # 修改为想要使用的模型类型
    
    # 创建实验实例并运行
    experiment = TrackingStabilityTask(subject_name, model_type)
    experiment.run() 