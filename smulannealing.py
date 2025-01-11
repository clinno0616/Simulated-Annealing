import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

# 设置绘图风格
plt.style.use('default')

# 生成随机城市坐标
np.random.seed(42)
n_cities = 20
cities = np.random.rand(n_cities, 2) * 100

# 计算两个城市之间的距离
def distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

# 计算路径总长度
def total_distance(route):
    return sum(distance(cities[route[i]], cities[route[i-1]]) for i in range(len(route)))

# 创建图形和子图
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(121)  # 路径图
ax2 = fig.add_subplot(122)  # 距离变化图

# 初始化数据
current_route = list(range(n_cities))
random.shuffle(current_route)
current_distance = total_distance(current_route)
distances = [current_distance]
iterations = [0]

# 算法参数
temperature = 100.0
cooling_rate = 0.995
min_temperature = 0.1
iteration = 0
best_distance = float('inf')
best_route = None

# 更新函数
def update(frame):
    global current_route, current_distance, temperature, iteration, best_distance, best_route
    
    # 清除当前图形
    ax1.clear()
    
    # 模拟退火迭代
    if temperature > min_temperature:
        # 生成新解
        new_route = current_route.copy()
        i, j = random.sample(range(n_cities), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_distance = total_distance(new_route)
        
        # 判断是否接受新解
        if new_distance < current_distance:
            current_route = new_route
            current_distance = new_distance
            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_route.copy()
        else:
            probability = math.exp((current_distance - new_distance) / temperature)
            if random.random() < probability:
                current_route = new_route
                current_distance = new_distance
        
        # 降温
        temperature *= cooling_rate
        
        # 记录数据
        distances.append(current_distance)
        iterations.append(iteration)
        iteration += 1
    
    # 绘制当前路径
    route_coords = np.array([cities[i] for i in current_route + [current_route[0]]])
    ax1.plot(route_coords[:, 0], route_coords[:, 1], 'b-', alpha=0.6)
    ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=100)
    ax1.set_title(f'Current Route\nDistance: {current_distance:.2f}\nBest Distance: {best_distance:.2f}')
    ax1.grid(True)
    
    # 绘制距离变化曲线
    ax2.clear()
    ax2.plot(iterations, distances, 'g-', alpha=0.6)
    ax2.set_title('Distance Over Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Distance')
    ax2.grid(True)
    
    # 设置图形范围和标题
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    plt.tight_layout()
    
    # 如果温度过低，停止动画
    if temperature <= min_temperature:
        anim.event_source.stop()
        plt.suptitle('Optimization Complete!', fontsize=16)
    else:
        plt.suptitle(f'Temperature: {temperature:.2f}', fontsize=16)

# 创建动画
anim = FuncAnimation(fig, update, frames=None, interval=50)
plt.show()