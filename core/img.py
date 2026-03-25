from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d


current_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = current_dir[:-4] + "file"


csv_files = sorted([f for f in os.listdir(file_dir) if f.endswith('.csv')])
Line = []
for i in csv_files:
    with open(os.path.join(file_dir, i)) as f:
        data = f.read()
        lines = data.splitlines()[17:]
        Line.extend(lines)
x = []
y = []
Line.sort()

def accelerate(list_x, list_y, smoothing_factor=3):
    """
    计算平滑的二阶导数
    smoothing_factor: 平滑因子，越大越平滑
    """
    list_x_acc = []
    list_y_acc = []
    
    
    smoothed_y = smooth_data(list_y, smoothing_factor)
    
    for i in range(1, len(list_x)-1):
        
        if list_x[i-1] == list_x[i] or list_x[i] == list_x[i+1]:
            continue
            
        # 使用三点公式计算二阶导数
        h1 = list_x[i] - list_x[i-1]
        h2 = list_x[i+1] - list_x[i]
        
        # 如果x间距不均匀，使用非均匀网格的二阶导数公式
        if abs(h1 - h2) < 1e-10:  # 近似相等
            h = h1
            second_deriv = (smoothed_y[i+1] - 2*smoothed_y[i] + smoothed_y[i-1]) / (h * h)
        else:
            # 非均匀网格的二阶导数公式
            second_deriv = (2/(h1*(h1+h2)) * smoothed_y[i-1] - 
                           2/((h1*h2)) * smoothed_y[i] + 
                           2/(h2*(h1+h2)) * smoothed_y[i+1])
        
        list_x_acc.append(list_x[i])
        list_y_acc.append(second_deriv)

    return list_x_acc, list_y_acc


def smooth_data(data, window_size):
    """
    使用移动平均对数据进行平滑处理
    """
    if window_size <= 1:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        window = data[start_idx:end_idx]
        smoothed.append(sum(window) / len(window))
    
    return smoothed


def derivative_first_order(list_x, list_y, smoothing_factor=3):
    """
    计算平滑的一阶导数
    smoothing_factor: 平滑因子，越大越平滑
    """
    list_x_deriv = []
    list_y_deriv = []
    
    # 首先对数据进行平滑处理
    smoothed_y = smooth_data(list_y, smoothing_factor)
    
    for i in range(1, len(list_x)-1):
        # 检查x值是否重复
        if list_x[i-1] == list_x[i] or list_x[i] == list_x[i+1]:
            continue
            
        # 使用中心差分法计算一阶导数：f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        h1 = list_x[i] - list_x[i-1]
        h2 = list_x[i+1] - list_x[i]
        
        # 如果x间距均匀，使用中心差分
        if abs(h1 - h2) < 1e-10:  # 近似相等
            h = (h1 + h2) / 2
            first_deriv = (smoothed_y[i+1] - smoothed_y[i-1]) / (2 * h)
        else:
            # 对于非均匀网格，使用三点公式
            # f'(x_i) ≈ [h2^2*f(x_{i-1}) - (h1+h2)*f(x_i) + h1^2*f(x_{i+1})] / (h1*h2*(h1+h2))
            first_deriv = ((smoothed_y[i+1] - smoothed_y[i]) / h2 + 
                          (smoothed_y[i] - smoothed_y[i-1]) / h1) / 2
        
        list_x_deriv.append(list_x[i])
        list_y_deriv.append(first_deriv)

    return list_x_deriv, list_y_deriv


def draw_x_y(list_x, list_y):
    plt.scatter(list_x, list_y, s=1)


def find_linear_region_with_second_derivative(x, y, greed_multiplier=0.1, voltage_percentile=0.7):
    """
    从电压高的一段开始选取点，选点区间满足二阶导的绝对值小于一个常数（贪心倍率），
    再在区间内掐头去尾，余下的点生成拟合直线
    """
    # 计算二阶导数
    second_deriv_x, second_deriv_y = accelerate(x, y, smoothing_factor=3)
    
    # 找到电压较高的起始点（例如，取电压最高的30%作为搜索范围）
    voltage_threshold = np.percentile(y, voltage_percentile * 100)
    high_voltage_indices = [i for i, voltage in enumerate(y) if voltage >= voltage_threshold]
    
    if not high_voltage_indices:
        print("未找到足够高的电压点")
        return None, None, None, None
    
    if len(second_deriv_x) < 2:
        print("二阶导数点太少")
        return None, None, None, None
    
    # 使用插值将二阶导数映射到原始x坐标
    try:
        # 确保x值单调递增用于插值
        sorted_pairs = sorted(zip(second_deriv_x, second_deriv_y))
        interp_second_deriv = interp1d([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], 
                                      kind='linear', bounds_error=False, fill_value=0)
        
        # 获取高电压区域对应的二阶导数值
        high_voltage_x = [x[i] for i in high_voltage_indices]
        high_voltage_second_deriv = [float(interp_second_deriv(xi)) for xi in high_voltage_x]
        
        # 找到满足二阶导数阈值的点
        valid_points = []
        for i, idx in enumerate(high_voltage_indices):
            if abs(high_voltage_second_deriv[i]) < greed_multiplier:
                valid_points.append(idx)
        
        if len(valid_points) < 10:  # 需要足够的点来进行后续处理
            print(f"找到的满足条件的点太少: {len(valid_points)}个")
            return None, None, None, None
        
        
        valid_points.sort(key=lambda idx: x[idx])
        
        # 掐头去尾
        trim_count = int(len(valid_points) * 0.7)
        trimmed_points = valid_points[trim_count : ]
        
        if len(trimmed_points) < 3:  
            print(f"掐头去尾后剩余点太少: {len(trimmed_points)}个")
            return None, None, None, None
        
        # 提取用于拟合的x和y值
        fit_x = [x[i] for i in trimmed_points]
        fit_y = [y[i] for i in trimmed_points]
        
        # 使用numpy进行线性拟合
        coefficients = np.polyfit(fit_x, fit_y, 1)
        slope, intercept = coefficients
        
        # 生成拟合直线上的点
        fit_line_x = np.array(fit_x)
        fit_line_y = slope * fit_line_x + intercept
        
        return fit_line_x, fit_line_y, slope, intercept
        
    except Exception as e:
        print(f"插值过程中出现错误: {e}")
        return None, None, None, None
    

for line in Line:
    values = line.split(',')
    y.append(float(values[0]))
    x.append(float(values[1]))
#draw_x_y(x, y)
# 绘制平滑的一阶导数
# draw_x_y(*derivative_first_order(x, y, smoothing_factor=3))
# 绘制平滑的二阶导数
# draw_x_y(*accelerate(x, y, smoothing_factor=5))

# 绘制原始数据
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=1, label='Tafel Curve Data')

# 寻找线性区域并拟合直线
fit_x, fit_y, slope, intercept = find_linear_region_with_second_derivative(x, y, greed_multiplier=0.1, voltage_percentile=0.7)

if fit_x is not None and fit_y is not None:
    # 计算整个x区间的拟合直线
    x_min, x_max = min(x), max(x)
    extended_x = np.linspace(x_min, x_max, 100)
    extended_y = slope * extended_x + intercept
    
    # 绘制延伸的拟合直线
    plt.plot(extended_x, extended_y, 'r-', label=f'Linear Fit: y = {slope:.6f}x + {intercept:.6f}', linewidth=2)
    
    # 标注找到的线性区域
    #plt.scatter(fit_x, fit_y, s=10, c='orange', label='Selected Linear Region Points', zorder=5)
    
    print(f"Fitted line equation: y = {slope:.6f}x + {intercept:.6f}")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
else:
    print("未能找到满足条件的线性区域")

plt.xlabel("Current (A)")
plt.ylabel("Voltage (V)")
plt.title("Tafel Curve with Linear Fit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(current_dir[:-4] + 'output/tafel_curve.png')