"""
项目绘图程序集成工具
利用opencv
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class MyDrawer:
    def __init__(self):
        self.width = 850
        self.height = 1200
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def background(self):
        # 坐标变换 x = x + 25; y = y + 45
        # 整体框架
        cv.rectangle(self.image, (25, 45), (825, 1150), (0, 0, 0), 1)
        # 后供料盘
        cv.rectangle(self.image, (25, 45), (825, 345), (0, 0, 0), 1)
        cv.line(self.image, (225, 45), (225, 345), (0, 0, 0), 1)
        cv.line(self.image, (425, 45), (425, 345), (0, 0, 0), 1)
        cv.line(self.image, (625, 45), (625, 345), (0, 0, 0), 1)
        # 治具盘
        cv.rectangle(self.image, (220, 360), (630, 720), (0, 0, 0), 1)
        # 底部相机
        cv.rectangle(self.image, (230, 735), (620, 835), (0, 0, 0), 1)
        cv.circle(self.image, (425, 785), 3, (0, 0, 0), -1)
        # 吸嘴站
        cv.rectangle(self.image, (640, 745), (800, 825), (0, 0, 0), 1)
        cv.circle(self.image, (720, 785), 3, (0, 0, 0), -1)
        # 前供料盘
        cv.rectangle(self.image, (25, 850), (825, 1150), (0, 0, 0), 1)
        cv.line(self.image, (225, 850), (225, 1150), (0, 0, 0), 1)
        cv.line(self.image, (425, 850), (425, 1150), (0, 0, 0), 1)
        cv.line(self.image, (625, 850), (625, 1150), (0, 0, 0), 1)

        # cv.putText(self.image, "Round 1", (370, 40),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv.imshow("background", self.image)

    def pickup(self, points):
        """绘制镍片的拾取过程"""
        for i in range(len(points)-1):
            cv.line(self.image, points[i], points[i+1], (0, 0, 255), 2)
            cv.circle(self.image, points[i], 3, (255, 0, 0), -1)
        cv.circle(self.image, points[-1], 3, (255, 0, 0), -1)
        cv.imshow("Pickup Process", self.image)

    def global_process(self, path):
        """绘制镍片贴装与拾取的整体路线图"""
        # 保存不同循环的路线图
        order = 0
        images = []
        img = self.image.copy()
        cv.circle(img, path[0], 3, (255, 0, 0), -1)
        for i in range(1, len(path) + 1):
            if i == len(path) or path[i] == -1:
                order = order + 1
                # 遇到循环分隔符，则新建画板
                # 先保存图像
                cv.putText(img, f"Round {order}", (370, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                # cv.imshow(f"第{order}轮拾取贴装", img)
                cv.imwrite(f"Round_{order}.png", img)
                images.append(img.copy())
                if i == len(path):
                    break
                img = self.image.copy()
                # 修正分隔符
                path[i] = path[i - 1]
            else:
                cv.line(img, path[i - 1], path[i], (0, 0, 255), 2)

            cv.circle(img, path[i], 3, (255, 0, 0), -1)




if __name__ == '__main__':
    # drawer = MyDrawer()
    # drawer.background()
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    a = [[0, 0], [0, 0], -1, [0, 0]]
    # for i in range(1, len(a)):
    #     print(a[i] == -1)
