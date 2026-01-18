# -*- coding: utf-8 -*-
"""
EUUP模糊度计算

该脚本实现了基于频率的模糊度计算，包括：
1. 从CSV文件中计算5分钟收益率
2. 生成频率分布
3. 计算模糊度（AMB）
4. 对多个文件夹进行批量处理
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_csv_files(folder_paths):
    """
    收集指定文件夹中的所有CSV文件路径
    
    Parameters:
    folder_paths (list): 文件夹路径列表
    
    Returns:
    list: 所有CSV文件的完整路径列表
    """
    print("开始收集CSV文件路径...")
    
    # 用于存储所有CSV文件的完整路径列表
    all_csv_file_paths = []
    
    # 遍历每个文件夹，收集所有CSV文件的完整路径
    for folder_path in folder_paths:
        # 确保文件夹存在，避免因为文件夹不存在而抛出异常
        if os.path.exists(folder_path):
            csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
            all_csv_file_paths.extend(csv_files)
        else:
            print(f"警告: 文件夹 {folder_path} 不存在")
    
    print(f"共收集到 {len(all_csv_file_paths)} 个CSV文件")
    return all_csv_file_paths


def calculate_5min_returns(selected_files):
    """
    计算5分钟收益率
    
    Parameters:
    selected_files (list): CSV文件路径列表
    
    Returns:
    dict: 包含每个文件收益率数据的字典
    """
    print("开始计算5分钟收益率...")
    
    # 使用字典来存储每个文件的收益率DataFrame 以及对应的日期      
    returns_dict = {}      
    
    # 使用tqdm来显示进度条    
    for file_path in tqdm(selected_files, desc="Processing files"):    
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
        
            # 创建收益率的DataFrame，并将日期列放在前面  
            returns_df = pd.DataFrame({'date': five_min_dates, '5min_return': five_min_returns})  
        
            # 将收益率DataFrame存储到字典中      
            returns_dict[filename] = returns_df      
        
        except Exception as e:      
            print(f"Error processing file {filename}: {e}")
    
    print("5分钟收益率计算完成")
    return returns_dict


def Freq_bin(r):
    """
    计算频率分布
    
    Parameters:
    r (array-like): 收益率数据
    
    Returns:
    array: 频率分布数组
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
    
    Parameters:
    returns_dict (dict): 包含收益率数据的字典
    
    Returns:
    dict: 包含每日分布的字典
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


def calculate_amb(daily_distributions, output_file):
    """
    计算模糊度（AMB）
    
    Parameters:
    daily_distributions (dict): 包含每日分布的字典
    output_file (str): 输出文件路径
    """
    print(f"开始计算模糊度并保存到 {output_file}...")
    
    # 初始化结果列表，用于存储code, date, 和amb  
    results = []  
    
    # 使用tqdm来创建一个进度条  
    # 注意：这里使用daily_distributions.items()的长度来确定进度条的总迭代次数  
    for code, daily_dist_list in tqdm(daily_distributions.items(), desc="Processing Codes", total=len(daily_distributions)):  
        # 遍历每日的分布列表  
        for date, freq_bin in daily_dist_list:  
            # 计算均值和方差（由于freq_bin是归一化的，均值可能接近0，但方差仍然有意义）  
            mean_bin = np.mean(freq_bin)  
            var_bin = np.var(freq_bin)  
            # 注意：原代码中对amb的计算可能存在问题，这里假设你想要的是方差的和  
            # 如果原意是别的，请根据实际需求调整  
            amb = np.sum(var_bin*mean_bin)  

            # 将code, date, 和amb添加到结果列表中  
            results.append([code, date, amb])  

    # 将结果保存为CSV文件  
    with open(output_file, 'w', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(['Code', 'Date', 'AMB'])  # 写入表头  
        csvwriter.writerows(results)  # 写入数据
    
    print(f"模糊度计算完成，结果已保存到 {output_file}")


def process_folder(folder_path, output_file):
    """
    处理单个文件夹的完整流程
    
    Parameters:
    folder_path (str): 输入文件夹路径
    output_file (str): 输出文件路径
    """
    print(f"开始处理文件夹: {folder_path}")
    
    # 1. 收集CSV文件
    folder_paths = [folder_path]
    selected_files = collect_csv_files(folder_paths)
    
    # 2. 计算5分钟收益率
    if selected_files:
        returns_dict = calculate_5min_returns(selected_files)
        
        # 3. 生成每日分布
        daily_distributions = generate_daily_distributions(returns_dict)
        
        # 4. 计算模糊度
        calculate_amb(daily_distributions, output_file)
    else:
        print(f"文件夹 {folder_path} 中没有找到CSV文件")
    
    print(f"文件夹 {folder_path} 处理完成")


def main():
    """
    主函数，执行完整的EUUP模糊度计算流程
    """
    print("开始执行EUUP模糊度计算流程...")
    
    # 定义要处理的文件夹和对应的输出文件
    folders_and_outputs = [
        ('新建文件夹1', 'amb_results1.csv'),
        ('新建文件夹2', 'amb_results2.csv'),
        ('新建文件夹3', 'amb_results3.csv'),
        ('新建文件夹4', 'amb_results4.csv'),
        ('新建文件夹5', 'amb_results5.csv'),
        ('新建文件夹6', 'amb_results6.csv'),
        ('新建文件夹7', 'amb_results7.csv'),
        ('新建文件夹8', 'amb_results8.csv'),
        ('新建文件夹9', 'amb_results9.csv')
    ]
    
    # 处理每个文件夹
    for folder_path, output_file in folders_and_outputs:
        process_folder(folder_path, output_file)
    
    print("EUUP模糊度计算流程完成！")


if __name__ == "__main__":
    main()