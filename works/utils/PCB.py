"""
PCB板的相关信息与设置
"""

import sys
from collections import Counter
import numpy as np
from sympy import false

point_range = [[220, 360], [630, 720]]  # 贴装点坐标范围，即贴装点位于治具盘范围内

class PCB:
    def __init__(self, num_point, num_type):
        self.num_point = num_point
        self.num_type = num_type

        # 定义供料盘中心定位
        self.feeder = [[125, 1000], [325, 1000], [525, 1000], [725, 1000],  # 前供料器
                       [125, 195], [325, 195], [525, 195], [725, 195]]      # 后供料器
        self.feeder = np.array(self.feeder)
        # 定义底部相机定位
        self.cam = [425, 785]
        # 定位换嘴站
        self.post = [720, 785]

        # 随机生成PCB的贴装点坐标
        x_label = np.random.randint(point_range[0][0], point_range[1][0]+1, size=50)
        y_label = np.random.randint(point_range[0][1], point_range[1][1]+1, size=50)
        self.point = np.column_stack((x_label, y_label))

        # 根据一定概率分配柔性盘
        prob = [2, 2.5, 2.5, 2, 1, 1.5, 1.5, 1]  # 各柔性盘的权重
        index_prob = list(enumerate(prob)) # 引入索引进行排序
        sort = sorted(index_prob, key=lambda x: (x[1], x[0]))[:8-num_type] # 舍去权重较低的若干个柔性盘
        for i, _ in sort:
            prob[i] = 0
        # 根据权重分配每个柔性盘中需要贴装的镍片数量
        prob = np.array(prob) / sum(prob)
        alloc = np.around(prob * num_point).astype(np.int8)
        max_index = np.argmax(alloc)
        alloc[max_index] = alloc[max_index] + (num_point - sum(alloc))
        self.alloc = alloc

        # 根据分配情况生成每个贴装点所贴装的贴片种类
        self.type = np.concatenate([np.full(count, value) for value, count in enumerate(self.alloc)])
        np.random.shuffle(self.type)


    def count(self, index):
        """统计指定区间所贴装的镍片数量"""
        return np.bincount(self.type[index], minlength=8)


class Nozzle:
    """定义一个吸嘴类"""
    def __init__(self):
        # 定义吸嘴的初始坐标
        x_label = np.random.randint(point_range[0][0], point_range[1][0]+1)
        y_label = np.random.randint(point_range[0][1], point_range[1][1]+1)
        self.p0 = [x_label, y_label]

        # 定义吸嘴种类
        self.num_type = 8
        # 定义吸嘴初始状态
        self.type = np.array(range(self.num_type))
        self.state = np.zeros(8).astype(np.uint8)

        # 定义贴装头的每个吸嘴到贴装头中心的距离
        self.dist = np.array([[-60, 50], [-20, 50], [20, 50], [60, 50],
                               [-60, -50], [-20, -50], [20, -50], [60, -50]])

    def count(self, with_state=False):
        """统计目前贴装头的各类吸嘴数量"""
        if with_state:
            count = np.zeros(8, dtype=int)
            for i in range(8):
                if self.state[i] == 1:
                    count[self.type[i]] = count[self.type[i]] + 1
            return count
        else:
            return np.bincount(self.type, minlength=8)

    def update(self, index, target):
        """根据需求，将对应位置的贴装杆换成对应所吸嘴"""
        # 可行性检查
        if len(index) != len(target):
            print("可更换吸嘴数量与需更换吸嘴数量不一致")
            sys.exit()

        self.type[index] = target
        



if __name__ == '__main__':
    pcb = PCB(50, 8)
    nozzle = Nozzle()
    print(nozzle.count())
    nozzle.type[3] = 0
    print(nozzle.count())
    # print(pcb.count())

