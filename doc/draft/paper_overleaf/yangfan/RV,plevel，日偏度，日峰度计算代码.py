# -*- coding: utf-8 -*-
"""
RV,plevel，日偏度，日峰度计算代码

该脚本实现了以下功能：
1. 收集指定文件夹中的CSV文件
2. 计算日波动率(RV)、偏度(Skewness)、峰度(Kurtosis)
3. 计算日内价格波动范围、换手率和日内价格反转
4. 计算plevel（价格水平）
5. 将结果保存到CSV文件
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


def calculate_rv_skew_kurt(df):
    """
    计算日波动率(RV)、偏度(Skewness)和峰度(Kurtosis)
    
    Parameters:
    df (DataFrame): 包含股票数据的DataFrame
    
    Returns:
    tuple: 包含RV、偏度、峰度、日内价格波动范围、换手率、日内价格反转和日期的列表
    """
    rv_values = []
    skew_values = []
    kurt_values = []
    intraday_range_values = []  # 日内价格波动范围
    turnover_rate_values = []  # 换手率
    intraday_reversal_values = []  # 日内价格反转
    rv_dates = []

    for i in range(0, len(df) - 239, 240):
        group = df.iloc[i:i + 240]
        returns = group['close'].pct_change().dropna()  # 计算收益率
        if len(returns) > 0:
            rv = returns.std()  # 计算日波动率
            skew = returns.skew()  # 计算偏度
            kurt = returns.kurt()  # 计算峰度（默认是超额峰度）
        else:
            rv = skew = kurt = np.nan

        # 计算日内价格波动范围
        if len(group) > 0:
            intraday_range = (group['high'].max() - group['low'].min()) / group['open'].iloc[0] if group['open'].iloc[0] != 0 else np.nan
        else:
            intraday_range = np.nan

        # 计算换手率
        if len(group) > 0:
            turnover_rate = group['amount'].sum() / group['value'].sum() if group['value'].sum() != 0 else np.nan
        else:
            turnover_rate = np.nan

        # 计算日内价格反转
        if len(group) > 0:
            mid_price = (group['high'].max() + group['low'].min()) / 2
            intraday_reversal = (group['close'].iloc[-1] - mid_price) / mid_price if mid_price != 0 else np.nan
        else:
            intraday_reversal = np.nan

        # 存储计算结果
        rv_values.append(rv)
        skew_values.append(skew)
        kurt_values.append(kurt)
        intraday_range_values.append(intraday_range)
        turnover_rate_values.append(turnover_rate)
        intraday_reversal_values.append(intraday_reversal)
        rv_dates.append(group['date'].iloc[0])

    return rv_values, skew_values, kurt_values, intraday_range_values, turnover_rate_values, intraday_reversal_values, rv_dates


def calculate_plevel(df):
    """
    计算plevel（价格水平）
    
    Parameters:
    df (DataFrame): 包含股票数据的DataFrame
    
    Returns:
    DataFrame: 添加了plevel列的DataFrame
    """
    df = df.copy()
    df['rank'] = df['close'].rank(method='first')
    df['plevel'] = df['rank'] / len(df)
    return df


def process_single_file(file_path):
    """
    处理单个CSV文件
    
    Parameters:
    file_path (str): CSV文件路径
    
    Returns:
    list: 处理后的数据列表
    """
    filename = os.path.basename(file_path)  # 获取文件名，不包含路径
    try:
        # 读取CSV文件到DataFrame
        df = pd.read_csv(file_path)

        # 验证必要的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'value', 'amount']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Required columns not found in {filename}")

        # 处理日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')  # 根据实际日期格式调整

        # 提取代码
        code = filename.split('.')[0]

        # 计算日波动率(RV)、偏度(Skewness)和峰度(Kurtosis)
        rv_values, skew_values, kurt_values, intraday_range_values, turnover_rate_values, intraday_reversal_values, rv_dates = calculate_rv_skew_kurt(df)

        # 计算plevel
        df = calculate_plevel(df)

        # 计算并存储plevel、RV、偏度、峰度、日内价格波动范围、换手率和日内价格反转
        saved_data = []
        rv_data = list(zip(rv_dates, rv_values, skew_values, kurt_values, intraday_range_values, turnover_rate_values, intraday_reversal_values))
        
        for i in range(0, len(df), 240):
            group = df.iloc[i:i + 240]
            if not group.empty:
                last_entry = group.iloc[-1]
                date = last_entry['date']
                plevel = last_entry['plevel']
                close = last_entry['close']

                # 查找对应的RV、偏度、峰度、日内价格波动范围、换手率和日内价格反转
                rv, skew, kurt, intraday_range, turnover_rate, intraday_reversal = None, None, None, None, None, None
                for rv_date, rv_value, skew_value, kurt_value, intraday_range_value, turnover_rate_value, intraday_reversal_value in rv_data:
                    if rv_date == date:
                        rv = rv_value
                        skew = skew_value
                        kurt = kurt_value
                        intraday_range = intraday_range_value
                        turnover_rate = turnover_rate_value
                        intraday_reversal = intraday_reversal_value
                        break

                # 存储数据
                saved_data.append({
                    'code': code,
                    'date': date,
                    'plevel': plevel,
                    'RV': rv,
                    'Skewness': skew,
                    'Kurtosis': kurt,
                    'Intraday_Range': intraday_range,  # 日内价格波动范围
                    'Turnover_Rate': turnover_rate,  # 换手率
                    'Intraday_Reversal': intraday_reversal  # 日内价格反转
                })

        return saved_data

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return []


def process_files(selected_files, output_file):
    """
    处理多个CSV文件并保存结果
    
    Parameters:
    selected_files (list): CSV文件路径列表
    output_file (str): 输出文件路径
    """
    print(f"开始处理 {len(selected_files)} 个文件...")
    
    # 使用列表来存储处理后的数据，最后合并为一个DataFrame
    all_data = []

    # 使用tqdm来显示进度条
    for file_path in tqdm(selected_files, desc="Processing files"):
        file_data = process_single_file(file_path)
        all_data.extend(file_data)

    # 将所有数据转换为DataFrame
    all_df = pd.DataFrame(all_data)

    # 保存为单个CSV文件
    all_df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")


def main():
    """
    主函数，执行完整的RV,plevel，日偏度，日峰度计算流程
    """
    print("开始执行RV,plevel，日偏度，日峰度计算流程...")
    
    # 设置要处理的文件夹路径
    folder_paths = ['新建文件夹5.1']  # 根据原笔记本设置
    
    # 收集CSV文件
    selected_files = collect_csv_files(folder_paths)
    
    # 处理文件并保存结果
    process_files(selected_files, 'all_data5(1).csv')
    
    print("RV,plevel，日偏度，日峰度计算流程完成！")


if __name__ == "__main__":
    main()