
import numpy as np

from utils.MyDrawer import MyDrawer
from utils.PCB import PCB, Nozzle
from sko.SA import SA_TSP
import cv2 as cv
import time

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
pickup_path = []    # 局部拾取路径
pickup_p0 = []      # 拾取过程起始点
global_path = []    # 全局路径存储变量

# 创建PCB对象
pcb = PCB(num_points, num_types)
# 创建Nozzle对象
nozzle = Nozzle()


############################### 测试代码 ##############################################
############################### 测试代码 ##############################################

############################### 准备工作 ##############################################
start = time.time()
def arrange_work():
    """根据整体贴装任务，为贴装过程进行规划"""
    # num_loop = 0    # 贴装循环总数
    plan_pick = []  # 保存每轮循环所贴装的各种镍片数量
    requests = np.copy(pcb.alloc)  # 备份每种镍片的贴装任务情况
    nozzle_num = nozzle.count()  # 保留吸嘴杆上的吸嘴数量
    flags = []  # 每一轮循环是否更换吸嘴的标志

    while sum(requests) != 0: # 循环直到所有贴装点均已完成
        flag = 0
        # num_loop = num_loop + 1 # 循环数增加
        need = requests - nozzle_num
        goal = nozzle_num  # 贴装目标与当前吸嘴数量一致
        if np.any(need < 0):  # 当变量中存在负数，则表示若不更换吸嘴则可能存在吸嘴空载
            if np.any(need > 0):  # 在当前吸嘴配置下，若剩下的贴装任务无法一次性完成，则进行更换吸嘴
                flag = 1
                change = np.where(need < 0)[0]  # 需要更换的吸嘴类型
                number = -sum(need[change])  # 可更换吸嘴数量
                index = np.array([], dtype=int)  # 保存可更换吸嘴所在杆索引
                for change_type in change:
                    change_num = -need[change_type]  # 该贴装头上该类吸嘴可更换的数量
                    options = np.where(nozzle.type == change_type)[0]
                    index = np.concatenate([index, options[-change_num:]])

                nozzle_num[change] = nozzle_num[change] + need[change]  # 卸载
                need[change] = 0  # 调整需求
                target_type = np.array([], dtype=int)  # 保存所更换的吸嘴类型
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
                        number = 0  # 更换吸嘴清空，结束循环
                    else:
                        # 当该种镍片的需求小于可更换吸嘴的数量，则根据需求换成相应所吸嘴
                        nozzle_num[max_index] = nozzle_num[max_index] + max_value
                        new = np.repeat(max_index, max_value)
                        target_type = np.concatenate([target_type, new])
                        number = number - max_value
                        need[max_index] = 0  # 该镍片需求清空，继续循环
            else:
                # 否则，贴装头上的吸嘴数量多余本轮的贴装任务，故贴装目标为剩余的贴装任务
                goal = requests
        flags.append(flag) # 保存变换标志
        plan_pick.append(goal.copy())
        requests = requests - goal

    return plan_pick, flags

plan_pick, flags = arrange_work()

def check_fix(new_route):
    """对每个路线进行检查与调整，令其符合贴装计划"""
    # new_route = route.copy()
    for i in range(len(plan_pick)):
        sec = (i+1)*8 if (i+1)*8 < len(new_route) else len(new_route)
        current = pcb.count(new_route[i*8:sec])
        goal = plan_pick[i]
        while True:
            gap = goal - current
            if not np.any(gap):
                break
            current_type = np.argmin(gap)
            goal_type = np.argmax(gap)
            temp = pcb.type[new_route]
            goal_loc = np.where(temp == goal_type)[0][-1]
            current_loc = np.where(temp[i*8:sec] == current_type)[0][0] + i*8

            # 交换元素
            t = new_route[goal_loc]
            new_route[goal_loc] = new_route[current_loc]
            new_route[current_loc] = t

            current[goal_type] = current[goal_type] + 1
            current[current_type] = current[current_type] - 1
    return new_route

def check_fix_chrom(chrom, size):
    """以种群为单位进行检验与修正"""
    for i in range(size):
        chrom[i, :] = check_fix(chrom[i, :])
    return chrom

print(f"准备工作完成，时间：{time.time() - start:.5f}秒")
############################### 拾取路径 ##############################################

def cal_distance(p1, p2):
    """计算两点间的距离"""
    return np.power((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2, 1/2)

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
            sa_pickup = SA_TSP(func=cal_part_distance, x0=range(num_pickup), T_max=t0, T_min=t1, L=10*num_pickup,
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

############################### 全局路径 ##############################################
## 定义功能函数
def cal_global_distance(route):
    """将一个路线个体装换为实际的贴装头路线，并得到全局路径长度"""

    global pickup_path, pickup_p0, global_path
    global_path = [nozzle.p0]  # 记录每轮贴装路径
    global_dist = 0
    # requests = np.copy(pcb.alloc)   # 备份每种镍片的贴装任务情况
    # nozzle_num = nozzle.count() # 保留吸嘴杆上的吸嘴数量

    for i in range(num_points):
        # order = order + 1;
        # 每轮贴装前进行拾取镍片
        if sum(nozzle.state) == 0:  # 根据贴装头上的负载情况判断上轮贴装是否完成
            # 重置拾取路径
            pickup_path = []

            pickup_p0 = global_path[-1]
            if not i==0:
                global_path.append(-1)  # 循环分隔符

            # # 判断是否需要更换吸嘴
            # flag = 0 # 更换吸嘴标志
            # need = requests - nozzle_num
            # goal = nozzle_num # 设置拾取目标
            # if np.any(need < 0): # 当变量中存在负数，则表示若不更换吸嘴则可能存在吸嘴空载
            #     if np.any(need > 0): # 在当前吸嘴配置下，若剩下的贴装任务无法一次性完成，则进行更换吸嘴
            #         flag = 1 # 更换吸嘴标志
            #         change = np.where(need < 0)[0] # 需要更换的吸嘴类型
            #         number = -sum(need[change]) # 可更换吸嘴数量
            #         index = np.array([], dtype=int) # 保存可更换吸嘴所在杆索引
            #         for change_type in change:
            #             change_num = -need[change_type] # 该贴装头上该类吸嘴可更换的数量
            #             options = np.where(nozzle.type == change_type)[0]
            #             index = np.concatenate([index, options[-change_num:]])
            #
            #     # if np.any(need > 0): # 在当前吸嘴配置下，若剩下的贴装任务无法一次性完成，则进行更换吸嘴
            #     #     flag = 1 # 更换吸嘴标志
            #         nozzle_num[change] = nozzle_num[change] + need[change] # 卸载
            #         need[change] = 0 # 调整需求
            #         target_type = np.array([], dtype=int) # 保存所更换的吸嘴类型
            #         while number > 0 and sum(need) != 0:
            #             # 贴装头上的空位且贴装需求不为零时进行安装吸嘴
            #             # 对贴装任务较多的镍片种类优先进行分配吸嘴
            #             max_value = np.max(need)
            #             max_index = np.argmax(need)
            #             if max_value >= number:
            #                 # 当该种镍片的需求大于可更换吸嘴的数量，则将吸嘴全换成该种镍片对应的吸嘴
            #                 nozzle_num[max_index] = nozzle_num[max_index] + number
            #                 new = np.repeat(max_index, number)
            #                 target_type = np.concatenate([target_type, new])
            #                 number = 0 # 更换吸嘴清空，结束循环
            #             else:
            #                 # 当该种镍片的需求小于可更换吸嘴的数量，则根据需求换成相应所吸嘴
            #                 nozzle_num[max_index] = nozzle_num[max_index] + max_value
            #                 new = np.repeat(max_index, max_value)
            #                 target_type = np.concatenate([target_type, new])
            #                 number = number - max_value
            #                 need[max_index] = 0 # 该镍片需求清空，继续循环
            #         nozzle.update(index, target_type) # 执行吸嘴更换操作
            #     else:
            #         # 修改拾取目标
            #         goal = requests


            plan = plan_pick[i//8]
            flag = flags[i//8]
            if flag == 1:
                nozzle.update(plan)

            # 到此为止，拾取过程已确定本轮拾取的镍片数量以及种类
            # 根据拾取目标goal进行镍片拾取工作
            for t, num in enumerate(plan):
                if num == 0:
                    continue
                # 查询该类吸嘴所在位置
                options = np.where(nozzle.type == t)[0]
                for opt in options:
                    # 坐标变换
                    location = pcb.feeder[t] - nozzle.dist[opt]
                    pickup_path.append(location.tolist())
                    # 更新吸嘴状态
                    nozzle.state[opt] = 1

            # 模拟退火优化拾取路线
            global_dist = global_dist + get_pickup_path(flag)
            global_path = global_path + pickup_path

        # 移动至下一个贴装点进行贴装
        # 贴装前先检查贴装头上该元件数量是否符合要求
        current_type = pcb.type[route[i]]
        option = np.where((nozzle.type == current_type) & (nozzle.state == 1))[0]
        if len(option) == 0:
            raise Exception("无效贴装")
        #     # 调整贴装顺序，以保证吸嘴尽可能满载
        #     left_num = sum(nozzle.state)
        #     current_type = np.argmax(nozzle.count(True) - pcb.count(route[i:i+left_num]))
        #     temp = pcb.type[route]
        #     # temp = np.array([pcb.type[i] for i in route])
        #     exchange = np.where(temp == current_type)[0][-1]
        #     # 交换贴装顺序
        #     temp = route[i]
        #     route[i] = route[exchange]
        #     route[exchange] = temp
        #
        #     option = np.where((nozzle.type == current_type) & (nozzle.state == 1))[0]

        # 移动至该贴装点
        opt = option[0]

        # 坐标变换
        new_point = pcb.point[route[i]] - nozzle.dist[opt]
        global_dist = global_dist + cal_distance(global_path[-1], new_point)
        global_path.append(new_point.tolist())
        # 更新吸嘴状态
        nozzle.state[opt] = 0
        # requests[current_type] = requests[current_type] - 1

    return global_dist

from sko.operators import ranking, selection, crossover, mutation

def crossover_udf(algorithm):
    """自定义交叉过程"""
    size_pop = algorithm.size_pop
    chrom = crossover.crossover_2point(algorithm)
    algorithm.Chrom = check_fix_chrom(chrom, size_pop)
    return algorithm.Chrom

def mutation_tsp(algorithm):
    """将同种镍片的贴装点顺序交换"""
    chrom = algorithm.Chrom
    for i in range(algorithm.size_pop):
        for j in range(algorithm.n_dim):
            if np.random.rand() < algorithm.prob_mut:
                current_type = pcb.type[chrom[i, j]]
                types = pcb.type[chrom[i,:]]
                options = np.where(types == current_type)[0]
                n = np.random.randint(0, len(options), 1)
                chrom[i, j], chrom[i, n] = chrom[i, n], chrom[i, j]
    algorithm.Chrom = chrom
    return algorithm.Chrom


def mutation_udf(algorithm):
    """
    自定义变异过程
    变异方式一共3中，每次随机选择其中一种
    """
    size_pop = algorithm.size_pop
    strategy = np.random.randint(3)
    if strategy == 0:
        """逆转变异"""
        chrom = mutation.mutation_reverse(algorithm)
        algorithm.Chrom = check_fix_chrom(chrom, size_pop)
    elif strategy == 1:
        """片段交换"""
        chrom = mutation.mutation_swap(algorithm)
        algorithm.Chrom = check_fix_chrom(chrom, size_pop)
    elif strategy == 2:
        algorithm.Chrom = mutation_tsp(algorithm)
    return algorithm.Chrom


############################### 拾取路径 ##############################################

route_test = np.array(range(num_points))
np.random.shuffle(route_test)
print(route_test)
route_test = check_fix(route_test)
print(f"修正工作完成，时间：{time.time() - start:.5f}秒")
# check_fix(route_test)
print(route_test)
print(cal_global_distance(route_test))
print(f"计算工作完成，时间：{time.time() - start:.5f}秒")
drawer.global_process(global_path)
print(f"绘制工作完成，时间：{time.time() - start:.5f}秒")
cv.waitKey(0)
cv.destroyAllWindows()