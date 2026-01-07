"""
两种方法计算模糊度的差尝试
分析AMBE和AMBMP两种模糊度指标的差异
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from matplotlib_venn import venn3


def load_data():
    """
    加载数据文件
    """
    # 定义文件列表
    file_names = ['all_data1.csv', 'all_data1(1).csv', 'all_data2.csv', 'all_data2(1).csv', 'all_data3.csv', 'all_data3(1).csv',
                  'all_data4.csv', 'all_data5.csv', 'all_data5(1).csv', 'all_data6.csv', 'all_data7.csv', 'all_data7(1).csv',
                  'all_data8.csv', 'all_data8(1).csv', 'all_data9.csv']

    # 初始化一个空列表来存储数据帧
    data_frames1 = []

    # 遍历文件列表，读取每个CSV文件并添加到列表中
    for file_name in file_names:
        # 读取CSV文件并添加到数据帧列表中
        data_frames1.append(pd.read_csv(file_name))

    # 使用pandas.concat将所有数据帧合并为一个DataFrame
    combined_dataframe1 = pd.concat(data_frames1, ignore_index=True)

    # 定义文件夹路径
    folder_path = '模糊度数据'

    # 初始化一个空列表来存储数据帧
    data_frames2 = []

    # 遍历文件夹下的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件扩展名是否为.csv
        if file_name.endswith('.csv'):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file_name)
            # 读取CSV文件并添加到数据帧列表中
            data_frames2.append(pd.read_csv(file_path))
    
    combined_dataframe2 = pd.concat(data_frames2, ignore_index=True)

    # 获取 combined_dataframe1 的列名集合
    cols1 = set(combined_dataframe1.columns)

    # 获取 combined_dataframe2 的列名集合，并找出独有的列
    cols2 = set(combined_dataframe2.columns)
    unique_cols2 = cols2 - cols1

    # 使用 pd.concat 合并 DataFrame，只包含 combined_dataframe2 中的独有列
    # axis=1 表示按列合并
    data = pd.concat([combined_dataframe1, combined_dataframe2[unique_cols2]], axis=1)
    data = data.drop(columns=["Date", "Code"])

    # 假设你的日期列是"Date"，并且格式为 datetime 类型
    # 如果不是 datetime 类型，你需要先将其转换为 datetime 类型
    data['date'] = pd.to_datetime(data['date'])

    # 定义一个函数，用于检查每个股票的最后日期是否满足条件
    def filter_stocks(group):
        # 获取该股票组内的最后日期
        last_date = group['date'].max()
        # 如果最后日期小于 2024-05-27，则返回 False，否则返回 True
        return last_date >= pd.Timestamp('2024-05-27')

    # 使用 groupby 和 filter 方法筛选满足条件的股票
    data = data.groupby('code').filter(filter_stocks)
    
    return data


def normalize_ambe(data):
    """
    对AMBE列进行归一化处理
    """
    # 获取AMBE列的长度
    ambe_length = len(data['AMBE'])

    # 初始化一个空列表来存储归一化后的AMBE值
    normalized_ambe = []

    # 设置进度条
    for i in tqdm(range(0, ambe_length, 48), desc="归一化进度"):
        # 获取当前段的AMBE数据
        segment = data['AMBE'][i:i+48]
        
        # 检查段长是否足够48，如果不足则跳过（可能是数据末尾不足48个）
        if len(segment) < 48:
            continue
        
        # 对当前段进行归一化处理
        min_val = segment.min()
        max_val = segment.max()
        normalized_segment = (segment - min_val) / (max_val - min_val)
        
        # 将归一化后的数据添加到列表中
        normalized_ambe.extend(normalized_segment.tolist())

    # 如果数据长度不是48的整数倍，处理剩余的数据（可选）
    remaining_data = data['AMBE'][ambe_length - (ambe_length % 48):]
    if len(remaining_data) > 0:
        min_val = remaining_data.min()
        max_val = remaining_data.max()
        normalized_remaining = (remaining_data - min_val) / (max_val - min_val)
        normalized_ambe.extend(normalized_remaining.tolist())

    # 将归一化后的AMBE列替换原数据中的AMBE列
    data['AMBE'] = normalized_ambe

    # 输出处理后的dataframe
    print(data)
    
    return data


def preprocess_data(df):
    """
    预处理步骤：替换inf和-inf为1，删除包含NaN的行，以及清洗AMBE和AMBMP的异常值
    """
    # 识别并删除包含无穷大、无穷小或NaN的行
    df = df.replace([np.inf, -np.inf], np.nan)  # 首先将inf和-inf替换为NaN
    df = df.dropna()  # 然后删除所有包含NaN的行
    
    # 清洗AMBMP列：删除不在0到1之间的值
    df = df[(df['AMBMP'] >= 0) & (df['AMBMP'] <= 1)]
    
    # 清洗AMBE列：采用四分位数间距（IQR）方法
    Q1 = df['AMBE'].quantile(0.25)
    Q3 = df['AMBE'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['AMBE'] >= lower_bound) & (df['AMBE'] <= upper_bound)]
    
    return df


def sort_data_by_plevel(data):
    """
    使用groupby根据'code'列分组，并对每组数据应用sort_values方法按'plevel'列降序排序
    """
    sorted_data = data.groupby('code').apply(lambda x: x.sort_values('plevel', ascending=False)).reset_index(drop=True)
    
    # 对排序后的数据进行预处理
    processed_data = preprocess_data(sorted_data)

    # 假设你的日期列是"date"，并且需要是 datetime 类型
    # 如果不是 datetime 类型，需要先将其转换为 datetime 类型
    processed_data['date'] = pd.to_datetime(processed_data['date'])

    # 定义一个函数，用于检查每个股票的最后日期是否满足条件
    def filter_stocks(group):
        # 获取该股票组内的最后日期
        last_date = group['date'].max()
        # 如果最后日期小于 2024-05-27，则返回 False，否则返回 True
        return last_date >= pd.Timestamp('2024-05-27')

    # 使用 groupby 和 filter 方法筛选满足日期条件的股票
    date_filtered_data = processed_data.groupby('code').filter(filter_stocks)

    # 计算 RV 列的分位数，例如计算第 50 个百分位数（中位数）作为筛选阈值
    rv_threshold = date_filtered_data['RV'].quantile(0.25)

    # 筛选出 RV 大于或等于阈值的股票数据
    filtered_data = date_filtered_data[date_filtered_data['RV'] >= rv_threshold]

    # 打印最终筛选后的数据
    print(filtered_data)
    
    return filtered_data


def plot_random_stocks(processed_data):
    """
    从plevel降序的数据中随机抽取4个股票进行可视化
    """
    # 设置matplotlib以显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 从plevel降序的数据中随机抽取4个股票的代码
    unique_codes = processed_data['code'].unique()
    random_codes = random.sample(list(unique_codes), 4)

    # 创建一个函数来计算20日均值
    def rolling_mean(series, window=30):
        return series.rolling(window=window).mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    colors = ['blue', 'green', 'red']  # 为AMB和收益率定义颜色
    line_styles = ['-', '--', '-.']  # 为AMB和收益率定义线型

    for i, code in enumerate(random_codes):
        stock_data = processed_data[processed_data['code'] == code]
        
        # 确保plevel是降序的
        stock_data = stock_data.sort_values(by='plevel', ascending=False)
        
        # 计算30日均值
        ambe_rolling = rolling_mean(stock_data['AMBE'])
        ambmp_rolling = rolling_mean(stock_data['AMBMP'])
        daily_log_return_rolling = rolling_mean(stock_data['daily_log_return'])
        
        # 画图（更新标签以反映30日均值）
        ax = axes[i]
        ax.plot(stock_data['plevel'], ambe_rolling, label='AMBE 30日均值', color=colors[0], linestyle=line_styles[1])
        ax.plot(stock_data['plevel'], ambmp_rolling, label='AMBMP 30日均值', color=colors[1], linestyle=line_styles[2])

        
        # 使用第二个y轴画日对数收益率
        ax2 = ax.twinx()
        ax2.plot(stock_data['plevel'], daily_log_return_rolling, label='Daily Log Return 30日均值', color=colors[2], linestyle=line_styles[0])
        
        # 设置图表标题和标签
        ax.set_title(f'股票代码: {code}')
        ax.set_xlabel('Plevel (降序)')
        ax.set_ylabel('AMB 30日均值')
        ax2.set_ylabel('Daily Log Return 30日均值')
        
        # 设置x轴的范围和刻度（如果plevel是从1到0的话）
        ax.set_xlim(1, 0)
        ax.set_xticks(np.arange(1, 0, -0.1))  # 这里设置了从1到0，每隔0.1一个刻度
        
        # 添加图例，确保图例不会重叠
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')  # 使用'best'自动选择最佳位置

    # 设置整体布局，确保图表不重叠
    plt.tight_layout(pad=3.0)

    # 显示图表
    plt.show()
    
    return random_codes


def plot_ambiguity_difference(processed_data, random_codes):
    """
    绘制模糊差值随plevel变化的折线图
    """
    # 设置matplotlib以显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建一个新的DataFrame来存储计算结果
    results = pd.DataFrame()

    # 计算每个股票的模糊差值，并添加到results DataFrame中
    for code in random_codes:
        stock_data = processed_data[processed_data['code'] == code]
        stock_data = stock_data.sort_values(by='plevel', ascending=False)
        stock_data['模糊差值'] = stock_data['AMBE'] - stock_data['AMBMP']
        results = pd.concat([results, stock_data[['code', 'plevel', '模糊差值']]], ignore_index=True)

    # 绘制模糊差值随plevel变化的折线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    colors = ['blue', 'green', 'red', 'purple']  # 为不同股票定义颜色

    for i, code in enumerate(random_codes):
        stock_data = results[results['code'] == code]
        
        # 画图
        ax = axes[i]
        ax.plot(stock_data['plevel'], stock_data['模糊差值'], label=f'股票代码: {code}', color=colors[i % len(colors)])
        
        # 设置图表标题和标签
        ax.set_title(f'股票代码: {code} - 模糊差值')
        ax.set_xlabel('Plevel (降序)')
        ax.set_ylabel('模糊差值')
        
        # 设置x轴的范围和刻度（如果plevel是从1到0的话）
        ax.set_xlim(1, 0)
        ax.set_xticks(np.arange(1, 0, -0.1))  # 这里设置了从1到0，每隔0.1一个刻度
        
        # 设置y轴的范围
        ax.set_ylim(-2, 2)
        
        # 添加图例
        ax.legend(loc='best')  # 使用'best'自动选择最佳位置

    # 设置整体布局，确保图表不重叠
    plt.tight_layout()

    # 显示图表
    plt.show()


def get_residuals(df):
    """
    定义一个函数来执行回归并返回下一期收益率的残差
    """
    X = df[['RV']]  # 使用上一期的RV作为自变量
    X = sm.add_constant(X)  # 添加常数项
    y = df['daily_log_return'].shift(-1)  # 使用下一期的收益率作为因变量
    valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
    X_valid = X[valid_rows]
    y_valid = y[valid_rows]
    model = RLM(y_valid, X_valid).fit()  # 使用RLM替代OLS
    residuals = pd.Series(model.resid, index=y_valid.index)
    return residuals.reindex(df.index)  # 用NaN填充没有残差的位置


def get_segment(start, segment_sizes):
    """
    获取当前段的名称
    """
    for segment, (start_idx, end_idx) in segment_sizes.items():
        if start >= start_idx and start < end_idx:
            return segment
    return None


def sliding_window_regression_with_segments(df, window_size, step, segment_sizes):
    """
    执行滑动窗口回归分析
    """
    segment_results = {segment: {'AMBE_p_value': [], 'AMBMP_p_value': [], 'AMBE_sign': [], 'AMBMP_sign': []} for segment in segment_sizes.keys()}
    
    for start in range(0, len(df) - window_size + 1, step):
        window_df = df.iloc[start:start + window_size]
        
        # 检查数据有效性
        if (window_df.shape[0] < window_size or 
            window_df[['AMBE', 'AMBMP', 'residual']].isnull().any().any() or
            window_df[['AMBE', 'AMBMP']].isin([np.inf, -np.inf]).any().any() or
            window_df['AMBE'].std() == 0 or window_df['AMBMP'].std() == 0):
            continue
        
        # 对AMBE进行回归
        X_ambe = window_df[['AMBE']]
        X_ambe = sm.add_constant(X_ambe)
        y = window_df['residual']
        model_ambe = RLM(y, X_ambe).fit()  # 使用RLM替代OLS
        ambe_p_value = model_ambe.pvalues['AMBE']
        segment_results[get_segment(start, segment_sizes)]['AMBE_p_value'].append(ambe_p_value)
        segment_results[get_segment(start, segment_sizes)]['AMBE_sign'].append(np.sign(model_ambe.params['AMBE']))
        
        # 对AMBMP进行回归
        X_ambmp = window_df[['AMBMP']]
        X_ambmp = sm.add_constant(X_ambmp)
        model_ambmp = RLM(y, X_ambmp).fit()  # 使用RLM替代OLS
        ambmp_p_value = model_ambmp.pvalues['AMBMP']
        segment_results[get_segment(start, segment_sizes)]['AMBMP_p_value'].append(ambmp_p_value)
        segment_results[get_segment(start, segment_sizes)]['AMBMP_sign'].append(np.sign(model_ambmp.params['AMBMP']))
    
    # 将结果转换为DataFrame
    result_dfs = {}
    for segment, results in segment_results.items():
        result_dfs[segment] = pd.DataFrame(results)
    
    return result_dfs


def perform_regression_analysis(filtered_data):
    """
    执行回归分析
    """
    # 定义窗口大小和步长
    window_size = 1000
    step = 30

    # 处理每个股票并收集结果
    unique_codes = filtered_data['code'].unique()
    all_segment_results = {}  # 用于存储所有结果

    for code in tqdm(unique_codes, desc="Processing stocks"):
        try:
            stock_df = filtered_data[filtered_data['code'] == code]
            stock_df = preprocess_data(stock_df)
            stock_df['residual'] = get_residuals(stock_df)
            
            # 为当前股票定义段的大小
            stock_length = len(stock_df) - window_size + 1
            if stock_length <= 0:
                continue
            segment_sizes = {
                'front': (0, stock_length // 3),
                'middle': (stock_length // 3, 2 * stock_length // 3),
                'back': (2 * stock_length // 3, stock_length)
            }
            
            # 执行滑动窗口回归并收集结果
            segment_results = sliding_window_regression_with_segments(stock_df, window_size, step, segment_sizes)
            all_segment_results[code] = segment_results
        
        except Exception as e:
            print(f"Error processing stock {code}: {e}")

    # 现在 all_segment_results 包含了所有股票的结果
    return all_segment_results


def analyze_good_stocks(all_segment_results):
    """
    分析表现良好的股票
    """
    # 筛选出表现良好的股票（基于p-value均值）
    good_stocks = set()
    for code, segment_results in all_segment_results.items():
        # 初始化p-value的累加器和计数器
        ambe_p_values = {'front': [], 'middle': [], 'back': []}
        ambmp_p_values = {'front': [], 'middle': [], 'back': []}
        
        # 收集每个段中的p-value
        for segment, results_df in segment_results.items():
            ambe_p_values[segment].extend(results_df['AMBE_p_value'])
            ambmp_p_values[segment].extend(results_df['AMBMP_p_value'])
        
        # 计算每个段中p-value的均值
        mean_ambe_p_values = {segment: np.mean(p_values) for segment, p_values in ambe_p_values.items()}
        mean_ambmp_p_values = {segment: np.mean(p_values) for segment, p_values in ambmp_p_values.items()}
        
        # 检查是否有任何一个段的均值p-value小于0.05
        if any(mean_p < 0.05 for mean_p in mean_ambe_p_values.values()) or any(mean_p < 0.05 for mean_p in mean_ambmp_p_values.values()):
            good_stocks.add(code)

    # 汇总表现良好的股票的回归结果
    good_all_p_values = {segment: {'AMBE': [], 'AMBMP': []} for segment in ['front', 'middle', 'back']}
    good_all_signs = {segment: {'AMBE': [], 'AMBMP': []} for segment in ['front', 'middle', 'back']}
    good_significant_counts = {segment: {'AMBE': 0, 'AMBMP': 0} for segment in ['front', 'middle', 'back']}
    good_total_counts = {segment: {'AMBE': 0, 'AMBMP': 0} for segment in ['front', 'middle', 'back']}

    for code in good_stocks:
        segment_results = all_segment_results[code]
        for segment, results_df in segment_results.items():
            good_all_p_values[segment]['AMBE'].extend(results_df['AMBE_p_value'])
            good_all_p_values[segment]['AMBMP'].extend(results_df['AMBMP_p_value'])
            good_all_signs[segment]['AMBE'].extend(results_df['AMBE_sign'])
            good_all_signs[segment]['AMBMP'].extend(results_df['AMBMP_sign'])
            
            good_significant_counts[segment]['AMBE'] += (results_df['AMBE_p_value'] < 0.05).sum()
            good_significant_counts[segment]['AMBMP'] += (results_df['AMBMP_p_value'] < 0.05).sum()
            
            good_total_counts[segment]['AMBE'] += len(results_df['AMBE_p_value'])
            good_total_counts[segment]['AMBMP'] += len(results_df['AMBMP_p_value'])

    # 计算显著比例
    good_significant_ratios = {segment: {metric: good_significant_counts[segment][metric] / good_total_counts[segment][metric] for metric in ['AMBE', 'AMBMP']} for segment in ['front', 'middle', 'back']}

    # 可视化 p-value 分布
    plt.figure(figsize=(14, 7))
    for segment in ['front', 'middle', 'back']:
        plt.subplot(1, 3, 1 + ['front', 'middle', 'back'].index(segment))
        sns.histplot(good_all_p_values[segment]['AMBE'], kde=True, label='AMBE', color='blue', alpha=0.5)
        sns.histplot(good_all_p_values[segment]['AMBMP'], kde=True, label='AMBMP', color='orange', alpha=0.5)
        plt.title(f'P-value Distribution in {segment.capitalize()} Segment (Good Stocks)')
        plt.xlabel('P-value')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.show()

    return good_stocks


def analyze_venn_diagram(all_segment_results):
    """
    分析Venn图
    """
    # 筛选出表现良好的股票（基于p-value）
    good_stocks = {
        'front': set(),
        'middle': set(),
        'back': set()
    }

    for code, segment_results in all_segment_results.items():
        for segment, results_df in segment_results.items():
            # 检查是否有任何一个p-value小于0.05
            if (results_df['AMBE_p_value'] < 0.05).any() or (results_df['AMBMP_p_value'] < 0.05).any():
                good_stocks[segment].add(code)

    # 输出每一段表现良好的股票
    for segment, stocks in good_stocks.items():
        print(f"Good stocks in {segment} segment: {sorted(stocks)}")

    # 计算两两段落之间的交集
    front_middle_intersection = good_stocks['front'].intersection(good_stocks['middle'])
    front_back_intersection = good_stocks['front'].intersection(good_stocks['back'])
    middle_back_intersection = good_stocks['middle'].intersection(good_stocks['back'])

    # 计算三个段落的交集
    all_three_intersection = good_stocks['front'].intersection(good_stocks['middle']).intersection(good_stocks['back'])

    # 计算三个段落的并集
    all_three_union = good_stocks['front'].union(good_stocks['middle']).union(good_stocks['back'])

    # 输出两两段落之间的交集
    print(f"Front & Middle Intersection: {sorted(front_middle_intersection)}")
    print(f"Front & Back Intersection: {sorted(front_back_intersection)}")
    print(f"Middle & Back Intersection: {sorted(middle_back_intersection)}")

    # 输出三个段落的交集
    print(f"Front & Middle & Back Intersection: {sorted(all_three_intersection)}")

    # 输出三个段落的并集
    print(f"Front & Middle & Back Union: {sorted(all_three_union)}")

    # 输出并集集合数量
    print(f"Number of unique good stocks in any segment: {len(all_three_union)}")

    # 可视化交集结果（Venn图）
    plt.figure(figsize=(10, 7))
    venn3([good_stocks['front'], good_stocks['middle'], good_stocks['back']], 
          ('Front', 'Middle', 'Back'), 
          alpha=0.5)
    plt.title("Venn Diagram of Good Stocks in Different Segments")
    plt.show()

    return good_stocks


def analyze_sign_patterns(segment_results, stock_list, segments=['front', 'middle', 'back']):
    """
    定义一个函数来分析每个子窗口的自变量符号规律
    """
    sign_patterns = {}
    for stock in stock_list:
        if stock in segment_results:
            stock_patterns = {}
            for segment in segments:
                if segment in segment_results[stock]:
                    results_df = segment_results[stock][segment]
                    # 获取自变量符号
                    ambe_signs = results_df['AMBE_sign']
                    ambmp_signs = results_df['AMBMP_sign']
                    # 计算主导符号
                    dominant_ambe_sign = np.sign(ambe_signs.mean())
                    dominant_ambmp_sign = np.sign(ambmp_signs.mean())
                    # 记录符号变化
                    sign_changes_ambe = (ambe_signs.diff().abs() > 0).sum()
                    sign_changes_ambmp = (ambmp_signs.diff().abs() > 0).sum()
                    # 存储结果
                    stock_patterns[segment] = {
                        'dominant_ambe_sign': dominant_ambe_sign,
                        'dominant_ambmp_sign': dominant_ambmp_sign,
                        'sign_changes_ambe': sign_changes_ambe,
                        'sign_changes_ambmp': sign_changes_ambmp,
                    }
            sign_patterns[stock] = stock_patterns
    return sign_patterns


def analyze_front_back_intersection(all_segment_results, good_stocks):
    """
    分析前后窗口交集的股票
    """
    # 筛选出前后窗口交集和三窗口交集的股票
    front_back_intersection = sorted(good_stocks['front'].intersection(good_stocks['back']))
    all_three_intersection = sorted(good_stocks['front'].intersection(good_stocks['middle']).intersection(good_stocks['back']))

    # 分析前后窗口交集的股票
    front_back_sign_patterns = analyze_sign_patterns(all_segment_results, front_back_intersection)

    # 分析三窗口交集的股票
    all_three_sign_patterns = analyze_sign_patterns(all_segment_results, all_three_intersection)

    # 输出分析结果
    print("Front & Back Intersection Sign Patterns:")
    for stock, patterns in front_back_sign_patterns.items():
        print(f"Stock {stock}:")
        for segment, pattern in patterns.items():
            print(f"  {segment} segment:")
            print(f"    Dominant AMBE sign: {pattern['dominant_ambe_sign']}")
            print(f"    Dominant AMBMP sign: {pattern['dominant_ambmp_sign']}")
            print(f"    AMBE sign changes: {pattern['sign_changes_ambe']}")
            print(f"    AMBMP sign changes: {pattern['sign_changes_ambmp']}")

    print("\nFront & Middle & Back Intersection Sign Patterns:")
    for stock, patterns in all_three_sign_patterns.items():
        print(f"Stock {stock}:")
        for segment, pattern in patterns.items():
            print(f"  {segment} segment:")
            print(f"    Dominant AMBE sign: {pattern['dominant_ambe_sign']}")
            print(f"    Dominant AMBMP sign: {pattern['dominant_ambmp_sign']}")
            print(f"    AMBE sign changes: {pattern['sign_changes_ambe']}")
            print(f"    AMBMP sign changes: {pattern['sign_changes_ambmp']}")

    return front_back_intersection, all_three_intersection


def results_to_dataframe(sign_patterns):
    """
    将分析结果转换为DataFrame
    """
    rows = []
    for stock, patterns in sign_patterns.items():
        for segment, pattern in patterns.items():
            rows.append({
                'Stock': stock,
                'Segment': segment,
                'Dominant AMBE Sign': pattern['dominant_ambe_sign'],
                'Dominant AMBMP Sign': pattern['dominant_ambmp_sign'],
                'AMBE Sign Changes': pattern['sign_changes_ambe'],
                'AMBMP Sign Changes': pattern['sign_changes_ambmp']
            })
    return pd.DataFrame(rows)


def analyze_differences(front_back_df, all_three_df):
    """
    分析差异
    """
    # 找出 front 和 back 主导正负号不同的股票及其具体作用效果
    diff_signs_ambe = []
    diff_signs_ambmp = []
    identical_signs = []
    same_effect_stocks = []  # 用于存储两种方法在前后窗口作用完全相同的股票
    
    for stock in front_back_df['Stock'].unique():
        front_row = front_back_df[(front_back_df['Stock'] == stock) & (front_back_df['Segment'] == 'front')]
        back_row = front_back_df[(front_back_df['Stock'] == stock) & (front_back_df['Segment'] == 'back')]
        
        if not front_row.empty and not back_row.empty:
            front_sign_ambe = front_row['Dominant AMBE Sign'].values[0]
            front_sign_ambmp = front_row['Dominant AMBMP Sign'].values[0]
            back_sign_ambe = back_row['Dominant AMBE Sign'].values[0]
            back_sign_ambmp = back_row['Dominant AMBMP Sign'].values[0]
            
            # 检查AMBE和AMBMP是否前后作用相反
            ambe_change = (front_sign_ambe != back_sign_ambe)
            ambmp_change = (front_sign_ambmp != back_sign_ambmp)
            
            # 记录不同的股票及其作用效果
            if ambe_change:
                diff_signs_ambe.append((stock, front_sign_ambe, back_sign_ambe))
            if ambmp_change:
                diff_signs_ambmp.append((stock, front_sign_ambmp, back_sign_ambmp))
            
            # 记录AMBE和AMBMP前后都相反的股票，并且变化趋势相同
            if ambe_change and ambmp_change and (front_sign_ambe == front_sign_ambmp) and (back_sign_ambe == back_sign_ambmp):
                identical_signs.append((stock, front_sign_ambe, back_sign_ambe, front_sign_ambmp, back_sign_ambmp))
            
            # 记录AMBE和AMBMP在前后窗口作用完全相同的股票
            if (front_sign_ambe == front_sign_ambmp) and (back_sign_ambe == back_sign_ambmp):
                same_effect_stocks.append((stock, front_sign_ambe, back_sign_ambe, front_sign_ambmp, back_sign_ambmp))
    
    # 格式化股票代码为六位
    diff_signs_ambe = [(f"{stock:06d}", front, back) for stock, front, back in diff_signs_ambe]
    diff_signs_ambmp = [(f"{stock:06d}", front, back) for stock, front, back in diff_signs_ambmp]
    identical_signs = [(f"{stock:06d}", front_ambe, back_ambe, front_ambmp, back_ambmp) for stock, front_ambe, back_ambe, front_ambmp, back_ambmp in identical_signs]
    same_effect_stocks = [(f"{stock:06d}", front_ambe, back_ambe, front_ambmp, back_ambmp) for stock, front_ambe, back_ambe, front_ambmp, back_ambmp in same_effect_stocks]

    # 创建集合A、集合B
    set_A = {stock for stock, _, _ in diff_signs_ambe}
    set_B = {stock for stock, _, _ in diff_signs_ambmp}

    # 找出A和B不同的股票
    A_non_B = list(set_A - set_B)
    non_A_B = list(set_B - set_A)
    A_and_B = list(set_A & set_B)

    # 准备输出结果
    output = ""
    
    # 集合A和集合B
    output += f"\n集合A（AMBE前后作用相反）:\n{sorted(set_A)}\n"
    output += f"\n集合B（AMBMP前后作用相反）:\n{sorted(set_B)}\n"
    
    # A非B
    output += f"\n\nA非B（只在集合A中，不在集合B中的股票）:\n{sorted(A_non_B)}\n"
    output += "\n详细A非B:\n"
    for stock, front, back in diff_signs_ambe:
        if stock in A_non_B:
            output += f"股票 {stock}: AMBE (front {front}, back {back})\n"
    
    # 非A B
    output += f"\n\n非A B（只在集合B中，不在集合A中的股票）:\n{sorted(non_A_B)}\n"
    output += "\n详细非A B:\n"
    for stock, front, back in diff_signs_ambmp:
        if stock in non_A_B:
            output += f"股票 {stock}: AMBMP (front {front}, back {back})\n"
    
    # A交B
    output += f"\n\nA交B（同时在集合A和集合B中的股票，即AMBE和AMBMP前后作用都相反的股票）:\n"
    for stock, front_ambe, back_ambe, front_ambmp, back_ambmp in identical_signs:
        if stock in A_and_B:
            output += f"股票 {stock}: AMBE (front {front_ambe}, back {back_ambe}), AMBMP (front {front_ambmp}, back {back_ambmp})\n"
    
    # 两种方法作用完全相同的股票
    output += f"\n\n两种方法作用完全相同的股票（前后窗口符号相同）:\n"
    for stock, front_ambe, back_ambe, front_ambmp, back_ambmp in same_effect_stocks:
        output += f"股票 {stock}: AMBE (front {front_ambe}, back {back_ambe}), AMBMP (front {front_ambmp}, back {back_ambmp})\n"
    
    # 计算总数和各种情况的比例
    total_stocks = len(front_back_intersection)
    A_count = len(set_A)
    B_count = len(set_B)
    A_non_B_count = len(A_non_B)
    non_A_B_count = len(non_A_B)
    A_and_B_count = len(A_and_B)
    same_effect_count = len(same_effect_stocks)
    
    output += f"\n\n总数: {total_stocks}"
    output += f"\nAMBE前后作用相反（集合A）的股票数: {A_count} ({A_count / total_stocks * 100:.2f}%)"
    output += f"\nAMBMP前后作用相反（集合B）的股票数: {B_count} ({B_count / total_stocks * 100:.2f}%)"
    output += f"\n只在集合A中的股票数: {A_non_B_count} ({A_non_B_count / total_stocks * 100:.2f}%)"
    output += f"\n只在集合B中的股票数: {non_A_B_count} ({non_A_B_count / total_stocks * 100:.2f}%)"
    output += f"\n同时在集合A和B中的股票数: {A_and_B_count} ({A_and_B_count / total_stocks * 100:.2f}%)"
    output += f"\n两种方法作用完全相同的股票数: {same_effect_count} ({same_effect_count / total_stocks * 100:.2f}%)"
    output += f"\n\n两种方法作用完全相同的股票（前后窗口符号相同）:\n"
    for stock, front_ambe, back_ambe, front_ambmp, back_ambmp in same_effect_stocks:
        output += f"股票 {stock}: AMBE (front {front_ambe}, back {back_ambe}), AMBMP (front {front_ambmp}, back {back_ambmp})\n"
    
    # 将输出结果保存到文本文件
    with open("输出结果.txt", "w", encoding="utf-8") as file:
        file.write(output)
    
    # 打印确认信息
    print("输出结果已保存到'输出结果.txt'文件中。")


def main_analysis():
    """
    主分析流程
    """
    print("开始加载数据...")
    data = load_data()
    print("数据加载完成")
    
    print("开始归一化AMBE...")
    data = normalize_ambe(data)
    print("AMBE归一化完成")
    
    print("开始数据预处理...")
    filtered_data = sort_data_by_plevel(data)
    print("数据预处理完成")
    
    print("生成描述性统计...")
    descriptive_statistics = filtered_data.describe()
    print(descriptive_statistics)
    
    print("统计股票代码的种类数...")
    num_unique_stocks = filtered_data['code'].nunique()
    print(f"股票代码种类数: {num_unique_stocks}")
    
    print("开始绘制随机股票图表...")
    random_codes = plot_random_stocks(data)
    print("随机股票图表绘制完成")
    
    print("开始绘制模糊差值图表...")
    plot_ambiguity_difference(data, random_codes)
    print("模糊差值图表绘制完成")
    
    print("开始回归分析...")
    all_segment_results = perform_regression_analysis(filtered_data)
    print("回归分析完成")
    
    print("分析表现良好的股票...")
    good_stocks = analyze_good_stocks(all_segment_results)
    print("表现良好的股票分析完成")
    
    print("分析Venn图...")
    good_stocks_sets = analyze_venn_diagram(all_segment_results)
    print("Venn图分析完成")
    
    print("分析符号模式...")
    front_back_intersection, all_three_intersection = analyze_front_back_intersection(all_segment_results, good_stocks_sets)
    print("符号模式分析完成")
    
    print("将结果转换为DataFrame...")
    front_back_df = results_to_dataframe(analyze_sign_patterns(all_segment_results, front_back_intersection))
    all_three_df = results_to_dataframe(analyze_sign_patterns(all_segment_results, all_three_intersection))
    print("结果转换完成")
    
    print("分析差异...")
    analyze_differences(front_back_df, all_three_df)
    print("差异分析完成")
    
    print("所有分析完成！")


if __name__ == "__main__":
    main_analysis()