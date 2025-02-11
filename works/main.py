
import numpy as np
from utils.MyDrawer import MyDrawer
from utils.PCB import PCB, Nozzle

# drawer = MyDrawer()
# drawer.background()
#
# cv.waitKey(0)
# cv.destroyAllWindows()


## 初始化参数
# 全局控制参数
num_points = 50 # 单次贴装任务的贴装点总数
num_types = 8   # 单次贴装任务的镍片种类数

# 为方便进行拾取优化，定义全局变量
pickup_path = []

# 创建PCB对象
pcb = PCB(num_points, num_types)
# 创建Nozzle对象
nozzle = Nozzle()


############################### 测试代码 ##############################################

# print(nozzle.p0)
# path = [nozzle.p0]
# pickup_path = nozzle.p0[:]
# pickup_path.append([0,0])
# print(nozzle.p0)
# nozzle_num = nozzle.count()
# print(nozzle.type)
# nozzle.type = np.array([0,1,2,7,4,5,6,7])
# print(nozzle.type)
# print(np.where(nozzle.type == 7)[0])
# a = np.array([0, 0])
# b = np.array([1, 1])
# c = [[2,2]]
# d = a+b
# c.append(d.tolist())
# print(c)
# route = []


############################### 测试代码 ##############################################
def cal_part_distance(route):
    """计算拾取过程的局部路径"""


def get_pickup_path(pickup_path, p0, p1, flag):
    """优化拾取路径"""
    return 1


## 定义功能函数
def routine2path(route):
    """将一个路线个体装换为实际的贴装头路线，并进行适当优化"""
    count_list = []  # 每轮贴装完成的贴装点数量
    order = 0  # 每轮贴装数量的计数变量
    global pickup_path
    path = [nozzle.p0]  # 记录每轮贴装路径
    requests = np.copy(pcb.alloc)   # 备份每种镍片的贴装任务情况
    nozzle_num = nozzle.count() # 保留吸嘴杆上的吸嘴数量

    for i in range(num_points):
        # order = order + 1;
        # 每轮贴装前进行拾取镍片
        if sum(nozzle.state) == 0:  # 根据贴装头上的负载情况判断上轮贴装是否完成
            # order = 1   # 新的一轮贴装循环，重置计算变量

            count = 0   # 本轮贴装循环所完成贴装点数

            # 确定拾取起始点
            p0 = path[-1]
            pickup_path = []

            # 确定拾取结束点
            p1 = pcb.point[i]

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
                    location = pcb.feeder[t] + nozzle.dist[opt]
                    pickup_path.append(location.tolist())

            pickup_path = get_pickup_path(pickup_path, p0, p1, flag)





            













def cal_global_distance(route):
    """计算单个贴装路线的贴装总距离"""



routine2path(route)