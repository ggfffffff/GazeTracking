import tkinter as tk
from tkinter import font as tkFont
import time

class TutorialApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("眼动控制教程")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.configure(bg="#f0f0f0")  # 使用浅灰色背景

        # 定义字体
        self.title_font = tkFont.Font(family="Microsoft YaHei", size=36, weight="bold")
        self.text_font = tkFont.Font(family="Microsoft YaHei", size=16)
        self.hint_font = tkFont.Font(family="Microsoft YaHei", size=14)

        # 创建画布用于动画效果
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg="#f0f0f0",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 初始化变量
        self.bottom_click_count = 0
        self.top_click_count = 0
        self.scale_var = tk.IntVar(value=100)

        self.setup_widgets()
        self.show_page1()

    def setup_widgets(self):
        # 页面1控件
        self.page1_title = self.canvas.create_text(
            self.screen_w/2, self.screen_h/4,
            text="欢迎使用眼动控制系统",
            font=self.title_font,
            fill="#2196F3",
            width=self.screen_w * 0.8
        )
        
        self.page1_desc = self.canvas.create_text(
            self.screen_w/2, self.screen_h/2,
            text=(
                "在本系统中，当您将视线停留在按钮上时，\n"
                "屏幕会出现一个圆圈渐渐填充，填满后即表示一次点击。\n"
                "您可以随时移动视线来取消或进行其他点击。"
            ),
            font=self.text_font,
            fill="#333333",
            width=self.screen_w * 0.6
        )

        # 创建开始按钮
        self.start_button = tk.Button(
            self.canvas,
            text="开始试用",
            font=self.text_font,
            bg="#2196F3",
            fg="white",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            command=self.show_page2
        )
        self.canvas.create_window(
            self.screen_w/2, self.screen_h * 0.75,
            window=self.start_button
        )

        # 页面2/3控件
        self.scale_bar = tk.Scale(
            self.canvas,
            from_=100, to=0,
            variable=self.scale_var,
            orient=tk.VERTICAL,
            length=self.screen_h * 0.8,
            bg="#f0f0f0",
            fg="#333333",
            font=self.text_font,
            troughcolor="#e0e0e0",
            highlightthickness=0
        )

        self.long_text = self.canvas.create_text(
            self.screen_w * 0.15, self.screen_h * 0.1,
            text=(
                "这是模拟的长文本内容。\n"
                "您可以想象这里有很多很多行，需要上下滚动查看。\n\n"
                "在真实场景中，你可以使用 Text 控件 + Scale 进行绑定，"
                "实现真正的滚动。\n"
                "此处只是展示布局和事件逻辑。\n\n"
                "在本教程里，我们用滑动条下端/上端操作来做练习。\n\n"
                "...\n(此处可继续添加大量文本)...\n\n"
                "让文本显得很长很长，需要用户拉动滑动条。"
            ),
            font=self.text_font,
            fill="#333333",
            width=self.screen_w * 0.4,
            anchor=tk.NW
        )

        # 提示文本
        self.page2_hint = self.canvas.create_text(
            self.screen_w * 0.45, self.screen_h * 0.9,
            text="注视滑动条的最下端可以下滑哦",
            font=self.hint_font,
            fill="#4CAF50"
        )

        self.page3_hint = self.canvas.create_text(
            self.screen_w * 0.45, self.screen_h * 0.1,
            text="注视滑动条的最上端可以上滑哦",
            font=self.hint_font,
            fill="#4CAF50"
        )

        # 页面4控件
        self.page4_text = self.canvas.create_text(
            self.screen_w * 0.9, self.screen_h * 0.1,
            text="恭喜你，已经学会眼动控制啦，\n接下来试试关掉这个教程吧！",
            font=self.text_font,
            fill="#F44336",
            anchor=tk.NE
        )

    def animate_transition(self, start_pos, end_pos, duration=0.5):
        start_time = time.time()
        while time.time() - start_time < duration:
            progress = (time.time() - start_time) / duration
            current_pos = start_pos + (end_pos - start_pos) * progress
            self.canvas.coords(self.long_text, self.screen_w * 0.15, current_pos)
            self.root.update()
        self.canvas.coords(self.long_text, self.screen_w * 0.15, end_pos)

    def show_page1(self):
        self.canvas.delete("all")
        self.setup_widgets()
        self.canvas.itemconfig(self.page1_title, state=tk.NORMAL)
        self.canvas.itemconfig(self.page1_desc, state=tk.NORMAL)
        self.start_button.pack()

    def show_page2(self):
        self.canvas.delete("all")
        self.setup_widgets()
        self.scale_bar.place(relx=0.03, rely=0.1)
        self.canvas.itemconfig(self.long_text, state=tk.NORMAL)
        self.canvas.itemconfig(self.page2_hint, state=tk.NORMAL)
        self.scale_var.set(100)
        self.scale_bar.config(command=self.on_scale_changed_page2)

    def show_page3(self):
        self.canvas.delete("all")
        self.setup_widgets()
        self.scale_bar.place(relx=0.03, rely=0.1)
        self.canvas.itemconfig(self.long_text, state=tk.NORMAL)
        self.canvas.itemconfig(self.page3_hint, state=tk.NORMAL)
        self.scale_var.set(0)
        self.scale_bar.config(command=self.on_scale_changed_page3)

    def show_page4(self):
        self.canvas.delete("all")
        self.setup_widgets()
        self.canvas.itemconfig(self.page4_text, state=tk.NORMAL)

    def on_scale_changed_page2(self, value):
        if value == 0:
            self.bottom_click_count += 1
            print(f"滑动条下端点击次数: {self.bottom_click_count}")
            if self.bottom_click_count >= 5:
                self.show_page3()

    def on_scale_changed_page3(self, value):
        if value == 100:
            self.top_click_count += 1
            print(f"滑动条上端点击次数: {self.top_click_count}")
            if self.top_click_count >= 5:
                self.show_page4()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TutorialApp()
    app.run()
