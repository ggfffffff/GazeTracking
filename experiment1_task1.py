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
import pyautogui

# 导入不同的眼动追踪模型
from gaze_tracking_basic import get_gaze_coordinates as get_gaze_basic
from gaze_tracking_coordinate import get_gaze_coordinates as get_gaze_coordinate
from gaze_tracking_kalman import get_gaze_coordinates as get_gaze_kalman, create_kalman_filter_acc
from gaze_tracking_multifilter import get_gaze_coordinates as get_gaze_multifilter, create_kalman_filter_acc as create_kf_multifilter
from gaze_tracking_optimized import get_gaze_coordinates as get_gaze_optimized, create_kalman_filter_acc as create_kf_optimized

class FixedPointTask:
    def __init__(self, subject_name, model_type="optimized"):
        self.screen_width = 1920
        self.screen_height = 1080
        self.target_radius = 20  # 目标点半径
        self.gaze_threshold = 100  # 注视判定阈值（像素）
        self.sample_duration = 5  # 采样持续时间（秒）
        self.interval_duration = 1  # 间隔时间（秒）
        self.num_targets = 5  # 目标点数量
        
        # 存储实验数据
        self.target_points = []  # 目标点坐标
        self.gaze_points = []    # 注视点坐标
        self.errors = []         # 误差数据
        
        # 创建结果目录
        self.results_dir = f"E1T1_{subject_name}_{model_type}"
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
        
        # 选择眼动追踪模型
        self.model_type = model_type
        self.initialize_model()
        
        # 初始化卡尔曼滤波器
        self.kf = self.create_kalman_filter_acc()
        self.initialized = False
        self.last_filtered = None
        self.recent_points = []
        self.last_stable_point = None
        self.last_update_time = time.time()
            
    def initialize_model(self):
        """初始化选择的眼动追踪模型"""
        if self.model_type == "basic":
            self.get_gaze_coordinates = lambda hr, vr: get_gaze_basic(hr, vr, self.calibration_data)
            self.kf = None
        elif self.model_type == "coordinate":
            self.get_gaze_coordinates = lambda hr, vr: get_gaze_coordinate(hr, vr, self.calibration_data)
            self.kf = None
        elif self.model_type == "kalman":
            self.kf = create_kalman_filter_acc()
            self.get_gaze_coordinates = lambda hr, vr: get_gaze_kalman(hr, vr, self.calibration_data, self.kf)
        elif self.model_type == "multifilter":
            self.kf = create_kf_multifilter()
            self.get_gaze_coordinates = lambda hr, vr: get_gaze_multifilter(hr, vr, self.calibration_data, self.kf)
        elif self.model_type == "optimized":
            self.kf = create_kf_optimized()
            self.get_gaze_coordinates = lambda hr, vr: get_gaze_optimized(hr, vr, self.calibration_data)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
        
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

        # 使用二次多项式拟合
        poly_x = np.polyfit(x_vals, screen_x_vals, 2)
        poly_y = np.polyfit(y_vals, screen_y_vals, 2)

        gaze_x = int(np.polyval(poly_x, hr))
        gaze_y = int(np.polyval(poly_y, vr)) + self.y_offset
        
        # 添加边界限制
        gaze_x = max(0, min(gaze_x, self.screen_width))
        gaze_y = max(0, min(gaze_y, self.screen_height))
        
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
                # 初始化卡尔曼滤波状态
                if not self.initialized:
                    self.kf.x = np.array([gaze_x, gaze_y, 0, 0, 0, 0])
                    self.last_filtered = (gaze_x, gaze_y)
                    self.initialized = True
                    
                # 卡尔曼预测 & 更新
                self.kf.predict()
                self.kf.update([gaze_x, gaze_y])
                kalman_point = (int(self.kf.x[0]), int(self.kf.x[1]))
                
                # 计算当前速度
                current_velocity = np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)
                
                # 低通滤波
                filtered_point = self.low_pass_filter(kalman_point, self.last_filtered,
                                                    alpha_x=0.2, alpha_y=0.1)
                
                # 添加移动平均
                self.recent_points.append(filtered_point)
                if len(self.recent_points) > self.moving_average_window:
                    self.recent_points.pop(0)
                averaged_point = self.moving_average(self.recent_points)
                
                # 防抖动处理
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                if current_velocity < self.min_velocity_threshold:
                    if self.last_stable_point is not None:
                        averaged_point = self.last_stable_point
                elif self.is_in_dead_zone(averaged_point, self.last_filtered):
                    averaged_point = self.last_filtered
                
                # 更新状态
                self.last_filtered = averaged_point
                self.last_stable_point = averaged_point
                self.last_update_time = current_time
                
                return averaged_point
        return None
            
    def generate_random_points(self):
        """生成随机目标点"""
        margin = 100  # 边缘留白
        self.target_points = []
        for _ in range(self.num_targets):
            x = np.random.randint(margin, self.screen_width - margin)
            y = np.random.randint(margin, self.screen_height - margin)
            self.target_points.append((x, y))
            
    def calculate_error(self, target, gaze):
        """计算注视点与目标点之间的误差"""
        return np.sqrt((target[0] - gaze[0])**2 + (target[1] - gaze[1])**2)
    
    def run_experiment(self):
        """运行实验"""
        if not self.calibration_data:
            print("未找到校准数据，请先运行校准程序。")
            return
            
        self.generate_random_points()
        
        # 创建全屏窗口
        cv2.namedWindow("Experiment", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for target_idx, target_point in enumerate(self.target_points):
            # 显示目标点
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(frame, target_point, self.target_radius, (0, 255, 0), -1)
            cv2.imshow("Experiment", frame)
            cv2.waitKey(1)
            
            # 等待用户注视并采样
            start_time = time.time()
            target_gaze_points = []
            
            while time.time() - start_time < self.sample_duration:
                # 创建新的帧
                frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                
                # 绘制目标点
                cv2.circle(frame, target_point, self.target_radius, (0, 255, 0), -1)
                
                # 获取并显示实时视线位置
                gaze_point = self.get_filtered_gaze_point()
                if gaze_point is not None:
                    cv2.circle(frame, gaze_point, 5, (0, 0, 255), -1)  # 红色小圆点表示视线位置
                    target_gaze_points.append(gaze_point)
                
                # 计算并显示倒计时
                remaining_time = int(self.sample_duration - (time.time() - start_time))
                if remaining_time > 0:
                    cv2.putText(frame, str(remaining_time), 
                              (self.screen_width//2 - 30, self.screen_height//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                
                cv2.imshow("Experiment", frame)
                cv2.waitKey(1)
                
            if target_gaze_points:
                # 计算平均注视点
                avg_gaze = np.mean(target_gaze_points, axis=0)
                self.gaze_points.append(avg_gaze)
                
                # 计算误差
                error = self.calculate_error(target_point, avg_gaze)
                self.errors.append(error)
            
            # 显示间隔
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.imshow("Experiment", frame)
            cv2.waitKey(int(self.interval_duration * 1000))
            
        cv2.destroyAllWindows()
        self.webcam.release()
        
    def calculate_metrics(self):
        """计算评价指标"""
        # 计算MAE
        mae = np.mean(self.errors)
        
        # 计算95%百分位误差
        percentile_95 = np.percentile(self.errors, 95)
        
        return {
            "MAE": mae,
            "95th_percentile_error": percentile_95
        }
    
    def generate_heatmap(self):
        """生成误差热力图"""
        # 创建网格
        x = np.linspace(0, self.screen_width, 50)
        y = np.linspace(0, self.screen_height, 50)
        X, Y = np.meshgrid(x, y)
        
        # 计算每个网格点的误差
        Z = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(Y)):
                point = (X[i,j], Y[i,j])
                errors = [self.calculate_error(point, gaze) for gaze in self.gaze_points]
                Z[i,j] = np.mean(errors)
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(Z, cmap='YlOrRd')
        plt.title('Error Heatmap')
        plt.savefig(os.path.join(self.results_dir, 'error_heatmap.png'))
        plt.close()
        
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 将numpy数组转换为普通列表
        avg_gaze_list = [gaze.tolist() if isinstance(gaze, np.ndarray) else gaze for gaze in self.gaze_points]
        
        results = {
            "timestamp": timestamp,
            "metrics": self.calculate_metrics(),
            "target_points": self.target_points,
            "gaze_points": avg_gaze_list,
            "errors": [float(err) for err in self.errors],  # 将numpy float转换为Python float
            "raw_data": {
                "target_points": self.target_points,
                "gaze_points": avg_gaze_list,
                "errors": [float(err) for err in self.errors]
            }
        }
        
        # 保存JSON格式的完整数据
        with open(os.path.join(self.results_dir, f'results_{timestamp}.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # 保存CSV格式的原始数据
        raw_data = np.column_stack((
            [p[0] for p in self.target_points],
            [p[1] for p in self.target_points],
            [p[0] for p in self.gaze_points],
            [p[1] for p in self.gaze_points],
            self.errors
        ))
        np.savetxt(
            os.path.join(self.results_dir, f'raw_data_{timestamp}.csv'),
            raw_data,
            delimiter=',',
            header='target_x,target_y,gaze_x,gaze_y,error',
            comments=''
        )
            
    def run(self):
        """运行完整的实验流程"""
        print("开始实验...")
        self.run_experiment()
        print("计算评价指标...")
        metrics = self.calculate_metrics()
        print(f"MAE: {metrics['MAE']:.2f}像素")
        print(f"95%百分位误差: {metrics['95th_percentile_error']:.2f}像素")
        print("生成热力图...")
        self.generate_heatmap()
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
    experiment = FixedPointTask(subject_name, model_type)
    experiment.run() 