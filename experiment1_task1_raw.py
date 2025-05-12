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

class FixedPointTaskRaw:
    def __init__(self, subject_name):
        self.screen_width = 1920
        self.screen_height = 1080
        self.target_radius = 20  # 目标点半径
        self.gaze_threshold = 100  # 注视判定阈值（像素）
        self.sample_duration = 2  # 采样持续时间（秒）
        self.interval_duration = 1  # 间隔时间（秒）
        self.num_targets = 5  # 目标点数量
        
        # 存储实验数据
        self.target_points = []  # 目标点坐标
        self.gaze_points = []    # 注视点坐标
        self.errors = []         # 误差数据
        
        # 创建结果目录
        self.results_dir = f"E1T1_{subject_name}_raw"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # 眼动追踪相关参数
        self.calibration_file = "calibration-5.json"
        self.y_offset = -10
        
        # 初始化眼动追踪
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.calibration_data = self.load_calibration()
            
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
                # 获取原始注视点
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                    
                self.gaze.refresh(frame)
                hr = self.gaze.horizontal_ratio()
                vr = self.gaze.vertical_ratio()
                
                if hr is not None and vr is not None:
                    gaze_x, gaze_y = self.get_gaze_coordinates(hr, vr)
                    if gaze_x is not None and gaze_y is not None:
                        target_gaze_points.append((gaze_x, gaze_y))
                
                time.sleep(0.01)  # 控制采样频率
                
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
        plt.title('Error Heatmap (Raw Gaze Tracking)')
        plt.savefig(os.path.join(self.results_dir, 'error_heatmap.png'))
        plt.close()
        
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "metrics": self.calculate_metrics(),
            "target_points": self.target_points,
            "gaze_points": self.gaze_points,
            "errors": self.errors
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
    # 在这里修改受试者姓名
    subject_name = "test_subject"
    experiment = FixedPointTaskRaw(subject_name)
    experiment.run() 