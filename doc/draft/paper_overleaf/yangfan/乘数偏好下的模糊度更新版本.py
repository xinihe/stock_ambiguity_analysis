"""
乘数偏好下的模糊度更新版本
基于频率的模糊度计算、聚类分析以及相关熵的计算
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


def load_csv_files(folder_paths):
    """
    加载指定文件夹中的所有CSV文件
    """
    all_csv_file_paths = []
    
    # 遍历每个文件夹，收集所有CSV文件的完整路径
    for folder_path in folder_paths:
        # 确保文件夹存在，避免因为文件夹不存在而抛出异常
        if os.path.exists(folder_path):
            csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
            all_csv_file_paths.extend(csv_files)
    
    return all_csv_file_paths


def process_stock_data(file_paths):
    """
    处理股票数据文件，计算5分钟收益率
    """
    returns_dict = {}
    
    # 使用tqdm来显示进度条
    for file_path in tqdm(file_paths, desc="Processing files"):
        filename = os.path.basename(file_path)  # 获取文件名，不包含路径
        try:
            # 读取CSV文件到DataFrame
            df = pd.read_csv(file_path)  # 如果CSV文件有标题行，则不需要names参数

            # 假设'close'列是收盘价，'date'列是日期，我们将用它们来计算收益率并处理日期格式
            if 'close' not in df.columns or 'date' not in df.columns:
                raise ValueError(f"Column 'close' or 'date' not found in {filename}")

            # 处理日期格式，将其转换为正确的日期类型，并创建一个新的日期列
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # 初始化一个空列表来存储5分钟收益率以及对应的日期
            five_min_returns = []
            five_min_dates = []

            # 按照固定的5个数据点一组来计算收益率，并同时记录对应的日期
            for i in range(0, len(df) - 4, 5):
                start_price = df['close'].iloc[i]
                end_price = df['close'].iloc[i + 4]
                return_val = (end_price - start_price) / start_price
                five_min_returns.append(return_val)
                # 记录中间日期（即这5个数据点的最后一个日期的次日，或者你可以根据需要选择其他日期）
                # 这里我们简单选择这组的结束日期作为代表日期
                five_min_dates.append(df['date'].iloc[i + 4])

            # 创建收益率的DataFrame，并将日期列放在前面，同时添加code列（这里假设code就是文件名）
            returns_df = pd.DataFrame({'code': [filename] * len(five_min_dates),
                                       'date': five_min_dates,
                                       '5min_return': five_min_returns})

            # 将收益率DataFrame存储到字典中
            returns_dict[filename] = returns_df

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return returns_dict


def Freq_bin(r):
    """
    将收益率映射到预定义的bins中，计算频率分布
    """
    # 将输入转换为numpy数组
    r = r.values if isinstance(r, pd.Series) else np.array(r)
    # 创建bins，范围从-20.1%到20.1%，共202个区间
    bin = np.linspace(-0.201, 0.201, 202)
    # 使用searchsorted将收益率映射到bins中，并进行边界裁剪
    r_bin = np.searchsorted(bin, r, side='right')  # 使用'right'来确保右闭区间
    r_bin = np.clip(r_bin, 0, 201)  # 裁剪到有效范围
    # 计算每个bin中的频率
    unique_values, counts = np.unique(r_bin, return_counts=True)
    freq_bin = np.zeros(202)
    freq_bin[unique_values - 1] = counts  # 减1来匹配bin的索引（针对左闭右开区间调整）
    # 归一化频率
    freq_bin = freq_bin / freq_bin.sum() if freq_bin.sum() > 0 else freq_bin
    return freq_bin


def generate_daily_distributions(returns_dict):
    """
    生成每日分布
    """
    daily_distributions = {}
    # 使用tqdm来显示整体进度
    for filename in tqdm(returns_dict, desc="Processing files"):
        returns_df = returns_dict[filename]
        if len(returns_df) >= 48:
            daily_dist_list = []
            # 使用tqdm来显示每个文件内部的进度
            for i in tqdm(range(0, len(returns_df), 48), desc=f"Processing {filename}", leave=False):
                group_returns = returns_df['5min_return'].iloc[i:i + 48]
                date = returns_df.iloc[i]['date']  # 提取48个收益率中的第一个日期
                freq_bin = Freq_bin(group_returns)
                daily_dist_list.append((date, freq_bin))  # 将日期和分布一起存储
            daily_distributions[filename] = daily_dist_list  # 存储每日的分布列表（包含日期）
        else:
            print(f"Not enough data for {filename} to generate a distribution.")
    return daily_distributions


def calculate_relative_entropy(stock_daily_dists, hs300_returns):
    """
    计算单个股票相对于沪深300的相对熵
    """
    # 使用seaborn绘制KDE并获取数据
    ax = sns.kdeplot(hs300_returns, bw_adjust=0.5, gridsize=1000)
    x, y = ax.lines[0].get_data()
    hs300_pdf = y / y.sum()  # 归一化以得到概率分布
    
    # 插值函数
    interp_func = interp1d(x, hs300_pdf, kind='linear', fill_value="extrapolate")
    
    # 只取前100天的分布
    stock_daily_dists_100 = stock_daily_dists[:50]
    
    # 初始化相对熵列表
    relative_entropies = []
    
    # 对前100天的每一天计算相对熵
    for date, stock_dist in stock_daily_dists_100:
        # 确保stock_dist是一个有效的概率分布
        stock_dist = np.maximum(stock_dist, 1e-10)
        stock_dist = stock_dist / stock_dist.sum()
        
        # 插值到hs300_pdf的网格上
        stock_dist_interp = interp_func(np.linspace(x.min(), x.max(), len(stock_dist)))
        
        # 计算相对熵
        rel_entropy = entropy(stock_dist, stock_dist_interp)
        relative_entropies.append(rel_entropy)
    
    return relative_entropies


def perform_clustering_analysis(all_relative_entropies):
    """
    对相对熵数据进行聚类分析
    """
    # 准备聚类数据
    cluster_data = {}
    for stock, relative_entropies in all_relative_entropies.items():
        if len(relative_entropies) >= 2:
            features = np.array([(relative_entropies[i-1], relative_entropies[i]) for i in range(1, len(relative_entropies))])
            print(f"Features for {stock}: {features.shape}")  # 调试输出
            cluster_data[stock] = features

    # 自适应KMeans聚类，但最大类别数不超过4
    kmeans_results = {}
    cluster_centers_y = {}  # 新增字典用于保存聚类中心的y值

    for stock, features in cluster_data.items():
        best_n_clusters = 1
        best_wss = np.inf
        wss_values = []
        
        for n_clusters in range(1, 5):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
            wss = kmeans.inertia_
            wss_values.append(wss)
            
            if wss < best_wss:
                best_wss = wss
                best_n_clusters = n_clusters
        
        # 使用最佳聚类数重新拟合KMeans
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # 保存聚类中心的y值
        cluster_centers_y[stock] = centers[:, 1]
        
        kmeans_results[stock] = {'labels': labels, 'centers': centers, 'features': features, 'n_clusters': best_n_clusters}
    
    return kmeans_results, cluster_centers_y


def calculate_ambiguity_metrics(daily_distributions, kmeans_results):
    """
    计算模糊度指标
    """
    # 假设的目标长度
    target_length = 202

    # 初始化一个空列表来收集所有结果
    results_list = []

    # 提取所有股票的代表分布中心和Current RE分布
    # （假设 kmeans_results 已经按照之前的逻辑被正确生成）
    representative_distributions = {}
    for stock, result in kmeans_results.items():
        cluster_current_re_distributions = {i: [] for i in range(result['n_clusters'])}
        for i, label in enumerate(result['labels']):
            current_re = result['features'][i][1]
            cluster_current_re_distributions[label].append(current_re)
        representative_distributions[stock] = {
            'centers': result['centers'],
            'current_re_distributions': cluster_current_re_distributions
        }

    # 初始化结果DataFrame
    results_df = pd.DataFrame(columns=['code', 'date', 'ambmp1'])

    # 计算每日分布与代表分布之间的JS divergence，并找到最接近的聚类分布
    for stock, daily_dists in daily_distributions.items():
        if stock in representative_distributions:
            stock_info = representative_distributions[stock]
            centers = stock_info['centers']
            current_re_distributions = stock_info['current_re_distributions']
            
            for date, daily_dist in daily_dists:
                # 确保daily_dist长度与目标长度一致
                if len(daily_dist) != target_length:
                    raise ValueError(f"Daily distribution length mismatch for stock {stock} on date {date}: expected {target_length}, got {len(daily_dist)}")
                
                # 将每日分布转换为概率分布
                daily_dist_prob = daily_dist / daily_dist.sum() if daily_dist.sum() > 0 else np.zeros_like(daily_dist)
                
                min_js = float('inf')
                closest_center = None
                closest_center_current_re = None
                
                for cluster_label, current_re_values in current_re_distributions.items():
                    if current_re_values:
                        current_re_prob = np.mean(current_re_values, axis=0)
                        current_re_prob = current_re_prob / current_re_prob.sum() if current_re_prob.sum() > 0 else np.zeros_like(current_re_prob)
                        
                        # 计算JS散度
                        js = jensenshannon(daily_dist_prob, current_re_prob)
                        
                        if js < min_js:
                            min_js = js
                            closest_center = centers[cluster_label]
                            closest_center_current_re = closest_center[1]  # 假设聚类中心的Current RE在第二个位置
                
                # 将结果添加到列表中
                if closest_center_current_re is not None:
                    results_list.append({
                        'code': stock,
                        'date': date,
                        'ambmp1': closest_center_current_re
                    })

    # 使用 pandas.DataFrame 将列表转换为 DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df


def calculate_ambiguity_with_js_divergence(daily_distributions):
    """
    使用JS散度计算模糊度
    """
    # 假设的目标长度
    target_length = 202
    n_clusters = 4

    # 初始化一个空字典来存储所有股票的聚类结果
    kmeans_results = {}

    # 定义一个函数来执行聚类分析并返回聚类中心和标签
    def perform_kmeans(distributions, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(distributions)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        return centers, labels

    # 使用tqdm包装主循环，处理每个股票的数据
    for stock in tqdm(daily_distributions.keys(), desc="Processing stocks"):
        daily_dists = daily_distributions[stock]
        
        # 准备用于聚类的特征：今天的相对熵与上一天的相对熵
        clustering_features = []
        for i in range(1, len(daily_dists)):
            today_dist = daily_dists[i][1]
            yesterday_dist = daily_dists[i-1][1]
            
            # 确保分布长度正确且非零
            if len(today_dist) == target_length and len(yesterday_dist) == target_length:
                if np.sum(today_dist) > 0 and np.sum(yesterday_dist) > 0:
                    today_prob = today_dist / np.sum(today_dist)
                    yesterday_prob = yesterday_dist / np.sum(yesterday_dist)
                    
                    # 计算相对熵（Jensen-Shannon divergence）
                    js_divergence = jensenshannon(today_prob, yesterday_prob)
                    clustering_features.append([js_divergence])
        
        # 如果有足够的特征进行聚类
        if len(clustering_features) > 10:
            centers, labels = perform_kmeans(np.array(clustering_features), n_clusters)
            kmeans_results[stock] = {
                'centers': centers,
                'features': clustering_features  # 保存用于聚类的特征
            }

    # 初始化一个空列表来收集所有结果
    results_list = []

    # 计算每日特征与聚类中心的最接近中心，并记录中心的纵坐标
    for stock, daily_dists in daily_distributions.items():
        if stock in kmeans_results:
            centers = kmeans_results[stock]['centers']
            
            for i in range(1, len(daily_dists)):
                today_dist = daily_dists[i][1]
                yesterday_dist = daily_dists[i-1][1]
                
                if len(today_dist) != target_length or len(yesterday_dist) != target_length:
                    continue
                if np.sum(today_dist) == 0 or np.sum(yesterday_dist) == 0:
                    continue
                
                today_prob = today_dist / np.sum(today_dist)
                yesterday_prob = yesterday_dist / np.sum(yesterday_dist)
                js_divergence = jensenshannon(today_prob, yesterday_prob)
                
                min_distance = float('inf')
                closest_center = None
                
                for center in centers:
                    distance = np.abs(center[0] - js_divergence)
                    if distance < min_distance:
                        min_distance = distance
                        closest_center = center
                
                # 使用聚类中心的纵坐标（即center[0]，因为我们是单特征聚类）
                center_value = closest_center[0] if closest_center is not None else np.nan
                results_list.append({
                    'code': stock,
                    'date': pd.to_datetime(daily_dists[i][0]),
                    'amb1': center_value
                })

    # 将结果转换为DataFrame并输出
    results_df = pd.DataFrame(results_list)
    
    return results_df


def load_hs300_data(file_path='daily_log_returns.csv'):
    """
    加载沪深300数据
    """
    hs300_daily_log_returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return hs300_daily_log_returns


def main_analysis(folder_paths=None):
    """
    主分析流程
    """
    if folder_paths is None:
        folder_paths = ['新建文件夹1', '新建文件夹1.1', '新建文件夹2', '新建文件夹2.2', '新建文件夹3', '新建文件夹3.1', '新建文件夹4', '新建文件夹5', '新建文件夹5.1', '新建文件夹6', '新建文件夹7', '新建文件夹7.1', '新建文件夹8', '新建文件夹8.1', '新建文件夹9']
    
    print("开始加载CSV文件...")
    all_csv_file_paths = load_csv_files(folder_paths)
    
    if len(all_csv_file_paths) < 4:
        print("CSV文件数量不足，无法随机选择4个文件。")
        selected_files = all_csv_file_paths
    else:
        # 随机选择4个文件
        selected_files = random.sample(all_csv_file_paths, min(4, len(all_csv_file_paths)))
        print("随机选择的文件是：")
        for file in selected_files:
            print(file)
    
    print("处理股票数据...")
    returns_dict = process_stock_data(selected_files)
    
    print("生成每日分布...")
    daily_distributions = generate_daily_distributions(returns_dict)
    
    print("加载沪深300数据...")
    hs300_daily_log_returns = load_hs300_data()
    
    print("计算相对熵...")
    all_relative_entropies = {}
    for stock, daily_dists in daily_distributions.items():
        # 计算单个股票的相对熵
        stock_relative_entropies = calculate_relative_entropy(daily_dists, hs300_daily_log_returns)
        # 存储结果
        all_relative_entropies[stock] = stock_relative_entropies
    
    print("进行聚类分析...")
    kmeans_results, cluster_centers_y = perform_clustering_analysis(all_relative_entropies)
    
    print("计算模糊度指标AMBMP1...")
    results_df_ambmp1 = calculate_ambiguity_metrics(daily_distributions, kmeans_results)
    print(results_df_ambmp1.head())
    
    print("使用JS散度计算模糊度指标AMB1...")
    results_df_amb1 = calculate_ambiguity_with_js_divergence(daily_distributions)
    print(results_df_amb1.head())
    
    # 保存结果
    results_df_ambmp1.to_csv('ambmp1_results.csv', index=False)
    results_df_amb1.to_csv('amb1_results.csv', index=False)
    
    print("分析完成！结果已保存到 ambmp1_results.csv 和 amb1_results.csv")
    
    return {
        'ambmp1_results': results_df_ambmp1,
        'amb1_results': results_df_amb1,
        'daily_distributions': daily_distributions,
        'kmeans_results': kmeans_results
    }


if __name__ == "__main__":
    # 设置matplotlib参数以支持中文显示
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'SimSun'  # SimSun为宋体
    
    results = main_analysis()