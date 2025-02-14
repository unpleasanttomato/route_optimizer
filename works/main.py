import random

import numpy as np
from utils.MyDrawer import MyDrawer
from utils.PCB import PCB, Nozzle
from sko.SA import SA_TSP
import cv2 as cv

drawer = MyDrawer()
drawer.background()
#
# cv.waitKey(0)
# cv.destroyAllWindows()


## 初始化参数
# 全局控制参数
num_points = 50 # 单次贴装任务的贴装点总数
num_types = 8   # 单次贴装任务的镍片种类数

# 为方便进行拾取优化，定义全局变量
pickup_path = []
pickup_p0 = []
global_path = []

# 创建PCB对象
pcb = PCB(num_points, num_types)
# 创建Nozzle对象
nozzle = Nozzle()


############################### 测试代码 ##############################################

pickup_p0 = nozzle.p0
pickup_path = [[125, 1000], [325, 1000], [525, 1000], [725, 1000],
               [125, 195], [325, 195], [525, 195], [725, 195]]
# pickup_path = [[125, 1000], [325, 1000]]
# pickup_path = [ [325, 1000]]
# print(pickup_path)
r = np.array(range(len(pickup_path)))
np.random.shuffle(r)

pickup_path = [pickup_path[i] for i in r]




# drawer.pickup(pickup_path)
# cv.waitKey(0)
# cv.destroyAllWindows()

############################### 测试代码 ##############################################
def cal_part_distance(route):
    """计算拾取过程的局部路径"""

    # 从拾取起始点到第一个柔性盘处
    len_part = np.power((pickup_p0[0] - pickup_path[route[0]][0])**2 +
                        (pickup_p0[1] - pickup_path[route[0]][1])**2, 1/2)
    # 拾取过程
    for i in range(len(route) - 1):
        len_part = len_part + np.power((pickup_path[route[i]][0] - pickup_path[route[i+1]][0])**2 +
                                       (pickup_path[route[i]][1] - pickup_path[route[i+1]][1])**2, 1/2)
    # 拾取完成后，到达底部相机
    len_part = len_part + np.power((pickup_path[route[-1]][0] - pcb.cam[0])**2 +
                                   (pickup_path[route[-1]][1] - pcb.cam[1])**2, 1/2)
    return len_part


def get_pickup_path(flag):
    """优化拾取路径"""
    global pickup_path, pickup_p0

    path = []
    dist = 0
    if flag == 1:
        # 当拾取前需要更换吸嘴，则将拾取起始点事先加入拾取路径
        # 并将起始点到吸嘴站的路径进行事先预存
        # 再将拾取起始点重置为吸嘴站
        dist = np.power((pickup_p0[0] - pcb.post[0])**2 +
                        (pickup_p0[1] - pcb.post[1])**2, 1/2)
        path.append(pcb.post)
        pickup_p0 = pcb.post

    num_pickup = len(pickup_path)
    # 根据问题规模设定温度
    if num_pickup > 6:
        t0 = 1e4
        t1 = 1e-4
    elif num_pickup > 4:
        t0 = 1e4
        t1 = 1e-2
    else:
        t0 = 1e2
        t1 = 1e-2

    if num_pickup == 1:
        # 当拾取元件仅为1时，拾取路径已确定
        # 依次经过起始点，柔性盘，底部相机和结束点
        path = path + pickup_path
        path.append(pcb.cam)
        # path.append(pickup_p1)
        dist = dist + cal_part_distance([0])
        pickup_path = path
    else:
        if num_pickup == 2:
            ans1 = [0, 1]
            ans2 = [1, 0]
            dist1, dist2 = cal_part_distance(ans1), cal_part_distance(ans2)
            best_path = ans1 if dist1 < dist2 else ans2
            best_dist = min(dist1, dist2)
        else:
            sa_pickup = SA_TSP(func=cal_part_distance, x0=range(num_pickup), T_max=t0, T_min=t1, L=20*num_pickup,
                               max_stay_counter=150)
            best_path, best_dist = sa_pickup.run()

            # 显示迭代过程
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(sa_pickup.best_y_history)
            # ax.set_xlabel("Iteration")
            # ax.set_ylabel("Distance")
            # plt.show()
        # print(best_path, best_dist)
        pickup_path = path + [pickup_path[i] for i in best_path]
        pickup_path.append(pcb.cam)
        dist = dist + best_dist

    return dist



# get_pickup_path(1)
#
# drawer.pickup(pickup_path)
# cv.waitKey(0)
# cv.destroyAllWindows()




## 定义功能函数
def routine2path(route):
    """将一个路线个体装换为实际的贴装头路线，并进行适当优化"""
    count_list = []  # 每轮贴装完成的贴装点数量
    order = 0  # 每轮贴装数量的计数变量
    global pickup_path, pickup_p0, global_path
    path = [nozzle.p0]  # 记录每轮贴装路径
    pickup_len = []
    requests = np.copy(pcb.alloc)   # 备份每种镍片的贴装任务情况
    nozzle_num = nozzle.count() # 保留吸嘴杆上的吸嘴数量

    for i in range(num_points):
        # order = order + 1;
        # 每轮贴装前进行拾取镍片
        if sum(nozzle.state) == 0:  # 根据贴装头上的负载情况判断上轮贴装是否完成
            # order = 1   # 新的一轮贴装循环，重置计算变量

            count = 0   # 本轮贴装循环所完成贴装点数

            # 重置拾取路径
            pickup_path = []

            pickup_p0 = path[-1]

            # 判断是否需要更换吸嘴
            flag = 0 # 更换吸嘴标志
            need = requests - nozzle_num
            if np.any(need < 0): # 当变量中存在负数，则表示若不更换吸嘴则可能存在吸嘴空载
                change = np.where(need < 0)[0] # 需要更换的吸嘴类型
                number = -sum(need[change]) # 可更换吸嘴数量
                index = np.array([], dtype=int) # 保存可更换吸嘴所在杆索引
                for change_type in change:
                    change_num = -need[change_type] # 该贴装头上该类吸嘴可更换的数量
                    options = np.where(nozzle.type == change_type)[0]
                    index = np.concatenate([index, options[-change_num:]])

                if np.any(need > 0): # 在当前吸嘴配置下，若剩下的贴装任务无法一次性完成，则进行更换吸嘴
                    flag = 1 # 更换吸嘴标志
                    nozzle_num[change] = nozzle_num[change] + need[change] # 卸载
                    need[change] = 0 # 调整需求
                    target_type = np.array([], dtype=int) # 保存所更换的吸嘴类型
                    while number > 0 and sum(need) != 0:
                        # 贴装头上的空位且贴装需求不为零时进行安装吸嘴
                        # 对贴装任务较多的镍片种类优先进行分配吸嘴
                        max_value = np.max(need)
                        max_index = np.argmax(need)
                        if max_value >= number:
                            # 当该种镍片的需求大于可更换吸嘴的数量，则将吸嘴全换成该种镍片对应的吸嘴
                            nozzle_num[max_index] = nozzle_num[max_index] + number
                            new = np.repeat(max_index, number)
                            target_type = np.concatenate([target_type, new])
                            number = 0 # 更换吸嘴清空，结束循环
                        else:
                            # 当该种镍片的需求小于可更换吸嘴的数量，则根据需求换成相应所吸嘴
                            nozzle_num[max_index] = nozzle_num[max_index] + max_value
                            new = np.repeat(max_index, max_value)
                            target_type = np.concatenate([target_type, new])
                            number = number - max_value
                            need[max_index] = 0 # 该镍片需求清空，继续循环
                    nozzle.update(index, target_type) # 执行吸嘴更换操作

            # 到此为止，拾取过程已确定本轮拾取的镍片数量以及种类
            # 根据nozzle_num进行镍片拾取工作
            for t, num in enumerate(nozzle_num):
                if num == 0:
                    continue
                # 查询该类吸嘴所在位置
                options = np.where(nozzle.type == t)[0]
                for opt in options:
                    # 坐标变换
                    location = pcb.feeder[t] + nozzle.dist[opt]
                    pickup_path.append(location.tolist())

            # 模拟退火优化拾取路线
            pickup_len.append(get_pickup_path(flag))





            













def cal_global_distance(route):
    """计算单个贴装路线的贴装总距离"""


