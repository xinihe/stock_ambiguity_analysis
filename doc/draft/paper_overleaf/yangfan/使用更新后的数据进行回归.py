# -*- coding: utf-8 -*-
"""
使用更新后的数据进行回归分析

该脚本实现了完整的金融数据分析流程，包括数据准备、预处理、
滚动回归、分价格区间回归、牛熊市分析和面板回归等。
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from linearmodels import PanelOLS
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from sklearn.preprocessing import MinMaxScaler


def load_and_merge_data():
    """
    加载并合并多个CSV数据文件
    """
    print("开始加载和合并数据...")
    
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
    
    print("已加载第一个数据集")
    
    # 从模糊度数据文件夹加载数据
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

    # 检查数据帧列表是否为空
    if data_frames2:
        # 使用pandas.concat将所有数据帧合并为一个DataFrame
        combined_dataframe2 = pd.concat(data_frames2, ignore_index=True)
        print("所有CSV文件已成功合并到一个DataFrame中")
    else:
        print("没有找到任何CSV文件来合并")
    
    # 获取 combined_dataframe1 的列名集合
    cols1 = set(combined_dataframe1.columns)

    # 获取 combined_dataframe2 的列名集合，并找出独有的列
    cols2 = set(combined_dataframe2.columns)
    unique_cols2 = cols2 - cols1

    # 使用 pd.concat 合并 DataFrame，只包含 combined_dataframe2 中的独有列
    # axis=1 表示按列合并
    data = pd.concat([combined_dataframe1, combined_dataframe2[unique_cols2]], axis=1)
    data = data.drop(columns=["Date", "Code"])

    print("数据合并完成")
    return data


def calculate_index_return():
    """
    计算沪深300指数的日收益率
    """
    print("计算沪深300指数日收益率...")
    
    # 读取CSV文件
    file_path = 'SSE.000300.csv'
    df = pd.read_csv(file_path)

    # 将datetime_nano列转换为DateTime类型
    df['datetime_nano'] = pd.to_datetime(df['datetime_nano'], format='%Y/%m/%d %H:%M:%S')

    # 初始化一个列表来存储对数收益率和对应的日期
    log_returns = []
    dates = []

    # 计算每240个数据点的对数收益率
    for i in range(0, len(df), 240):
        # 获取当前片段的结束索引，确保不超出数据范围
        end_index = min(i + 240, len(df))
        
        # 获取当前片段的第一个和最后一个收盘价
        first_close = df['SSE.000300.close'].iloc[i]
        last_close = df['SSE.000300.close'].iloc[end_index - 1]
        
        # 计算对数收益率
        log_return = np.log(last_close / first_close)
        log_returns.append(log_return)
        
        # 获取当前片段的结束日期，并仅保留年月日
        end_date = df['datetime_nano'].iloc[end_index - 1].date()  # 使用 .date() 方法
        dates.append(end_date)

    # 创建一个DataFrame来存储结果
    result_df = pd.DataFrame({
        'Date': dates,
        'Log Return': log_returns
    })

    # 如果需要，可以将结果保存到新的CSV文件
    result_df.to_csv('沪深300日收益率.csv', index=False)
    
    print("沪深300指数日收益率计算完成")
    return result_df


def preprocess_data(df):
    """
    预处理数据：替换inf和-inf为NaN，删除包含NaN的行
    """
    # 识别并删除包含无穷大、无穷小或NaN的行
    df = df.replace([np.inf, -np.inf], np.nan)  # 首先将inf和-inf替换为NaN
    df = df.dropna()  # 然后删除所有包含NaN的行
    
    return df


def filter_stocks_by_date(df):
    """
    根据日期条件过滤股票
    """
    print("根据日期条件过滤股票...")
    
    # 假设你的日期列是"date"，并且格式为 datetime 类型
    # 如果不是 datetime 类型，你需要先将其转换为 datetime 类型
    df['date'] = pd.to_datetime(df['date'])

    # 定义一个函数，用于检查每个股票的最后日期是否满足条件
    def filter_stocks(group):
        # 获取该股票组内的最后日期
        last_date = group['date'].max()
        # 如果最后日期小于 2024-05-27，则返回 False，否则返回 True
        return last_date >= pd.Timestamp('2024-05-27')

    # 使用 groupby 和 filter 方法筛选满足条件的股票
    filtered_data = df.groupby('code').filter(filter_stocks)
    
    print("股票过滤完成")
    return filtered_data


def calculate_excess_return(filtered_data, result_df):
    """
    计算超额收益
    """
    print("计算超额收益...")
    
    # 将 filtered_data 的 'date' 列转换为 DateTime 类型
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])

    # 确保 result_df 的索引是 DateTime 类型（假设 'date' 是日期列）
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.set_index('Date', inplace=True)
        result_df.index = pd.to_datetime(result_df.index)

    # 创建一个新的列来存储超额收益
    filtered_data['Excess Return'] = np.nan

    # 使用 tqdm 为 iterrows 添加进度条
    for idx, row in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0], desc="Processing Rows"):
        stock_date = row['date']
        # 检查 stock_date 是否在 result_df 的索引中
        if stock_date in result_df.index:
            index_return = result_df.loc[stock_date, 'Log Return']  # 确保列名正确
            stock_return = row['daily_log_return']
            filtered_data.at[idx, 'Excess Return'] = stock_return - index_return

    # 删除 'Excess Return' 列中的缺失值
    filtered_data = filtered_data.dropna(subset=['Excess Return'])
    
    print("超额收益计算完成")
    return filtered_data


def sliding_window_regression_with_segments(df, window_size, step, segment_sizes):
    """
    滑动窗口回归分析
    """
    def get_segment(start, segment_sizes):
        for segment, (start_idx, end_idx) in segment_sizes.items():
            if start >= start_idx and start < end_idx:
                return segment
        return None
    
    def get_residuals(df):
        X = df[['RV']]  # 确保这里的列名也是小写
        X = sm.add_constant(X)  # 添加常数项
        y = df['daily_log_return'].shift(-1)  # 使用下一期的收益率作为因变量
        valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
        X_valid = X[valid_rows]
        y_valid = y[valid_rows]
        model = RLM(y_valid, X_valid).fit()  # 使用RLM替代OLS
        residuals = pd.Series(model.resid, index=y_valid.index)
        return residuals.reindex(df.index)  # 用NaN填充没有残差的位置

    segment_results = {segment: {'ambe_p_value': [], 'ambe_sign': []} for segment in segment_sizes.keys()}
    
    for start in range(0, len(df) - window_size + 1, step):
        window_df = df.iloc[start:start + window_size]
        
        # 检查数据有效性
        if (window_df.shape[0] < window_size or 
            window_df[['AMBE', 'residual']].isnull().any().any() or
            window_df['AMBE'].isin([np.inf, -np.inf]).any() or
            window_df['AMBE'].std() == 0):
            continue
        
        # 对AMBE进行回归
        X_ambe = window_df[['AMBE']]
        X_ambe = sm.add_constant(X_ambe)
        y = window_df['residual']
        model_ambe = RLM(y, X_ambe).fit()  # 使用RLM替代OLS
        ambe_p_value = model_ambe.pvalues['AMBE']
        segment_results[get_segment(start, segment_sizes)]['ambe_p_value'].append(ambe_p_value)
        segment_results[get_segment(start, segment_sizes)]['ambe_sign'].append(np.sign(model_ambe.params['AMBE']))
    
    # 将结果转换为DataFrame
    result_dfs = {}
    for segment, results in segment_results.items():
        result_dfs[segment] = pd.DataFrame(results)
    
    return result_dfs


def analyze_rolling_regression(data):
    """
    执行滚动回归分析
    """
    print("开始滚动回归分析...")
    
    def preprocess_data(df):
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 1)
        return df
    
    def get_residuals(df):
        X = df[['RV']]  # 确保这里的列名也是小写
        X = sm.add_constant(X)  # 添加常数项
        y = df['daily_log_return'].shift(-1)  # 使用下一期的收益率作为因变量
        valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
        X_valid = X[valid_rows]
        y_valid = y[valid_rows]
        model = RLM(y_valid, X_valid).fit()  # 使用RLM替代OLS
        residuals = pd.Series(model.resid, index=y_valid.index)
        return residuals.reindex(df.index)  # 用NaN填充没有残差的位置

    def sliding_window_regression_with_segments(df, window_size, step, segment_sizes):
        def get_segment(start, segment_sizes):
            for segment, (start_idx, end_idx) in segment_sizes.items():
                if start >= start_idx and start < end_idx:
                    return segment
            return None
        
        segment_results = {segment: {'ambe_p_value': [], 'ambe_sign': []} for segment in segment_sizes.keys()}
        
        for start in range(0, len(df) - window_size + 1, step):
            window_df = df.iloc[start:start + window_size]
            
            # 检查数据有效性
            if (window_df.shape[0] < window_size or 
                window_df[['AMBE', 'residual']].isnull().any().any() or
                window_df['AMBE'].isin([np.inf, -np.inf]).any() or
                window_df['AMBE'].std() == 0):
                continue
            
            # 对AMBE进行回归
            X_ambe = window_df[['AMBE']]
            X_ambe = sm.add_constant(X_ambe)
            y = window_df['residual']
            model_ambe = RLM(y, X_ambe).fit()  # 使用RLM替代OLS
            ambe_p_value = model_ambe.pvalues['AMBE']
            segment_results[get_segment(start, segment_sizes)]['ambe_p_value'].append(ambe_p_value)
            segment_results[get_segment(start, segment_sizes)]['ambe_sign'].append(np.sign(model_ambe.params['AMBE']))
        
        # 将结果转换为DataFrame
        result_dfs = {}
        for segment, results in segment_results.items():
            result_dfs[segment] = pd.DataFrame(results)
        
        return result_dfs

    # 定义窗口大小和步长
    window_size = 1000
    step = 30

    # 处理每个股票并收集结果
    unique_codes = data['code'].unique()
    all_segment_results = {}  # 用于存储所有结果

    for code in tqdm(unique_codes, desc="Processing stocks"):
        try:
            stock_df = data[data['code'] == code]
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
    
    print("滚动回归分析完成")
    return all_segment_results


def get_residuals(df):
    """
    获取残差
    """
    X = df[['RV']]  # 确保这里的列名也是小写
    X = sm.add_constant(X)  # 添加常数项
    y = df['daily_log_return'].shift(-1)  # 使用下一期的收益率作为因变量
    valid_rows = ~X.isnull().any(axis=1) & ~y.isnull()
    X_valid = X[valid_rows]
    y_valid = y[valid_rows]
    model = RLM(y_valid, X_valid).fit()  # 使用RLM替代OLS
    residuals = pd.Series(model.resid, index=y_valid.index)
    return residuals.reindex(df.index)  # 用NaN填充没有残差的位置


def analyze_by_price_levels(data):
    """
    按价格水平进行分组回归分析
    """
    print("开始按价格水平进行分组回归分析...")
    
    def preprocess_data(df):
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 1)
        return df
    
    # 使用groupby根据'code'列分组，并对每组数据应用sort_values方法按'plevel'列降序排序
    sorted_data = data.groupby('code').apply(lambda x: x.sort_values('plevel', ascending=False)).reset_index(drop=True)
    
    def regress_segment(segment_df, segment_name, results, significance_level=0.05):
        if segment_df.empty or segment_df[['AMBE_lag', 'daily_log_return']].isnull().any().any() or segment_df['AMBE_lag'].std() == 0:
            return
        
        X_ambe = segment_df[['AMBE_lag']]
        X_ambe = sm.add_constant(X_ambe)
        y = segment_df['Excess Return']
        model_ambe = RLM(y, X_ambe).fit()
        ambe_p_value = model_ambe.pvalues['AMBE_lag']
        results[segment_name]['ambe_p_value'].append(ambe_p_value)
        results[segment_name]['ambe_sign'].append(np.sign(model_ambe.params['AMBE_lag']))
        results[segment_name]['significant'] = (ambe_p_value < significance_level) or results[segment_name]['significant']  # 保持显著状态

    def process_stock(code, stock_df, segment_results):
        try:
            stock_df = preprocess_data(stock_df)
            
            # 计算 AMBE 的滞后一期
            stock_df['AMBE_lag'] = stock_df['AMBE'].shift(1)
            
            stock_df_sorted = stock_df.sort_values(by='plevel')
            
            n = len(stock_df_sorted)
            segment_size = n // 3
            
            front_df = stock_df_sorted.iloc[:segment_size]
            middle_df = stock_df_sorted.iloc[segment_size:2*segment_size]
            back_df = stock_df_sorted.iloc[2*segment_size:]
            
            regress_segment(front_df, 'front', segment_results)
            regress_segment(middle_df, 'middle', segment_results)
            regress_segment(back_df, 'back', segment_results)
            
        except Exception as e:
            print(f"Error processing stock {code}: {e}")

    # 处理每个股票并收集结果
    unique_codes = sorted_data['code'].unique()
    all_segment_results = {}
    significant_results = []  # 用于存储显著结果

    for code in tqdm(unique_codes, desc="Processing stocks"):
        stock_df = sorted_data[sorted_data['code'] == code]
        segment_results = {
            'front': {'ambe_p_value': [], 'ambe_sign': [], 'significant': False},
            'middle': {'ambe_p_value': [], 'ambe_sign': [], 'significant': False},
            'back': {'ambe_p_value': [], 'ambe_sign': [], 'significant': False}
        }
        all_segment_results[code] = segment_results
        process_stock(code, stock_df, segment_results)
        
        # 检查并保存显著结果
        for segment, result in segment_results.items():
            if result['significant']:
                significant_results.append({
                    'code': code,
                    'segment': segment,
                    'ambe_coefficient': result['ambe_sign'][0] if result['ambe_sign'] else None,  # 假设只有一个值
                    'significant': result['significant']
                })

    # 将显著结果保存为 CSV 文件
    significant_df = pd.DataFrame(significant_results)
    significant_df.to_csv('分价格回归结果1.csv', index=False)
    
    print("按价格水平分组回归分析完成")
    return significant_df


def analyze_by_price_levels_regression(filtered_data):
    """
    按不同价位进行回归分析
    """
    print("开始按不同价位进行回归分析...")
    
    # 确保数据按股票代码和时间排序
    filtered_data = filtered_data.sort_values(by=['code', 'date'])

    # 对 AMBE 进行滞后一期处理
    filtered_data['AMBE_lag1'] = filtered_data.groupby('code')['AMBE'].shift(1)

    # 计算 plevel 的分位数
    plevel_q25 = filtered_data['plevel'].quantile(0.25)
    plevel_q75 = filtered_data['plevel'].quantile(0.75)

    # 根据 plevel 的分位数将数据分为三组
    low_plevel_data = filtered_data[filtered_data['plevel'] <= plevel_q25]
    medium_plevel_data = filtered_data[(filtered_data['plevel'] > plevel_q25) & (filtered_data['plevel'] <= plevel_q75)]
    high_plevel_data = filtered_data[filtered_data['plevel'] > plevel_q75]

    # 定义函数来运行回归
    def run_regression(data):
        y = data['daily_log_return']
        X = data[['AMBE_lag1', 'RV','Turnover_Rate', 'Intraday_Range', 'Skewness', 'Kurtosis']]
        model = PanelOLS(dependent=y, exog=X, entity_effects=True, time_effects=True, drop_absorbed=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        return results

    # 分别运行低 plevel 组、中 plevel 组和高 plevel 组的回归
    low_plevel_results = run_regression(low_plevel_data)
    medium_plevel_results = run_regression(medium_plevel_data)
    high_plevel_results = run_regression(high_plevel_data)

    # 定义函数来提取结果并保存为DataFrame
    def extract_results(results, group_name):
        params = results.params
        pvalues = results.pvalues
        conf_int = results.conf_int()
        results_df = pd.DataFrame({
            'Group': [group_name] * len(params),
            'Variable': params.index,
            'Coefficient': params.values,
            'P-value': pvalues.values,
            'Lower CI': conf_int['lower'].values,
            'Upper CI': conf_int['upper'].values
        })
        return results_df

    # 提取并保存三个组的结果
    low_plevel_df = extract_results(low_plevel_results, 'Low Plevel')
    medium_plevel_df = extract_results(medium_plevel_results, 'Medium Plevel')
    high_plevel_df = extract_results(high_plevel_results, 'High Plevel')
    
    # 输出结果
    print("Low Plevel Group Results:")
    print(low_plevel_df)
    print("\nMedium Plevel Group Results:")
    print(medium_plevel_df)
    print("\nHigh Plevel Group Results:")
    print(high_plevel_df)
    
    return low_plevel_df, medium_plevel_df, high_plevel_df


def analyze_market_conditions(filtered_data):
    """
    分析牛熊市条件
    """
    print("分析牛熊市条件...")
    
    # 定义牛市和熊市的时间段
    bull_markets = [
        ('1990-12-19', '1992-05-26'),
        ('1992-11-17', '1993-02-16'),
        ('1994-07-29', '1994-09-13'),
        ('1996-01-19', '1997-05-12'),
        ('1999-05-19', '2001-06-14'),
        ('2005-06-06', '2007-10-16'),
        ('2008-10-28', '2009-08-04'),
        ('2013-06-25', '2015-06-12'),  # 或使用（'2014-03-12', '2015-06-12'）
        ('2019-01-04', '2021-02-18')
    ]

    bear_markets = [
        ('1992-05-26', '1992-11-17'),
        ('1993-02-16', '1994-07-29'),
        ('1994-09-13', '1995-02-07'),
        ('1995-05-22', '1996-01-19'),
        ('1997-05-12', '1999-05-18'),
        ('2001-06-14', '2005-06-06'),
        ('2007-10-16', '2008-10-28'),
        ('2009-08-04', '2013-06-25'),
        ('2015-06-12', '2016-01-27'),
        ('2018-01-29', '2019-01-04'),
        ('2021-02-18', '2025-01-15')  # 假设至今的日期到2025-01-15
    ]

    # 将字符串日期转换为日期对象
    bull_markets = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bull_markets]
    bear_markets = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bear_markets]

    # 初始化牛市和熊市的虚拟变量列
    filtered_data['bull_market'] = 0
    filtered_data['bear_market'] = 0

    # 标记牛市
    for start, end in bull_markets:
        mask = (filtered_data['date'] >= start) & (filtered_data['date'] <= end)
        filtered_data.loc[mask, 'bull_market'] = 1

    # 标记熊市
    for start, end in bear_markets:
        mask = (filtered_data['date'] >= start) & (filtered_data['date'] <= end)
        filtered_data.loc[mask, 'bear_market'] = 1

    # 定义板块分组函数
    def get_stock_market(code):
        """
        根据股票代码判断所属板块。
        - 主板：以 600、601、603、605、000、001 开头
        - 中小板：以 002 开头
        - 创业板：以 300、301 开头
        - 科创板：以 688 开头
        """
        code_str = str(code).zfill(6)  # 补齐 6 位
        if code_str.startswith(('600', '601', '603', '605', '000', '001')):
            return '主板'
        elif code_str.startswith('002'):  # 中小板
            return '中小板'
        elif code_str.startswith(('300', '301')):  # 创业板
            return '创业板'
        elif code_str.startswith('688'):  # 科创板
            return '科创板'
        else:
            return '未知板块'

    # 添加板块列
    filtered_data['market'] = filtered_data['code'].apply(get_stock_market)
    
    print("牛熊市条件分析完成")
    return filtered_data


def run_market_based_panel_regression(filtered_data):
    """
    运行基于牛熊市的面板回归分析
    """
    print("开始基于牛熊市的面板回归分析...")
    
    # 确保数据按股票代码和时间排序
    filtered_data = filtered_data.sort_values(by=['code', 'date'])

    # 对 AMBE 进行滞后一期处理
    filtered_data['AMBE_lag1'] = filtered_data.groupby('code')['AMBE'].shift(1)
    
    # 创建交互项
    filtered_data['bull_market_AMBE'] = filtered_data['bull_market'] * filtered_data['AMBE']
    filtered_data['bear_market_AMBE'] = filtered_data['bear_market'] * filtered_data['AMBE']

    # 1. 对 RV 进行中心化
    rv_mean = filtered_data['RV'].mean()
    filtered_data['RV_centered'] = filtered_data['RV'] - rv_mean

    # 2. 生成中心化后的 RV 二次项
    filtered_data['RV_centered_squared'] = filtered_data['RV_centered'] ** 2

    # 3. 生成中心化后的交互项
    filtered_data['AMBE_lagged_RV_centered'] = filtered_data['AMBE_lag1'] * filtered_data['RV_centered']
    filtered_data['AMBE_lagged_RV_centered_squared'] = filtered_data['AMBE_lag1'] * filtered_data['RV_centered_squared']

    # 转换为面板格式（MultiIndex）
    filtered_data = filtered_data.set_index(['code', 'date'])

    # 因变量
    y = filtered_data['daily_log_return']  # 确保这里使用的是超额收益

    # 自变量与控制变量
    exog_vars = ['AMBE_lag1', 'bull_market', 'bear_market', 'bull_market_AMBE', 'bear_market_AMBE', 'RV', 'Turnover_Rate', 'Intraday_Range', 'Skewness', 'Kurtosis']  # 根据新模型调整变量列表
    X = filtered_data[exog_vars]

    # 运行双向固定效应模型（此处仅控制个体固定效应，可根据需要调整）
    # 设定模型（个体固定效应）
    model = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,  # 控制个体固定效应
        time_effects=False,   # 如果需要时间固定效应，可以设置为 True
        drop_absorbed=True    # 自动删除被吸收的变量
    )

    # 拟合模型（使用双向聚类标准误，可根据需要选择其他协方差类型）
    results = model.fit(cov_type='clustered', cluster_entity=True)  # 如果需要时间聚类，可以添加 cluster_time=True

    # 提取关键参数
    params = results.params
    pvalues = results.pvalues
    conf_int = results.conf_int()

    # 将结果保存为 DataFrame
    results_df = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'P-value': pvalues.values,
        'Lower CI': conf_int['lower'].values,
        'Upper CI': conf_int['upper'].values
    })

    # 打印结果
    print("基于牛熊市的面板回归分析完成")
    print(results_df)
    
    return results_df


def run_panel_regression(filtered_data):
    """
    运行面板回归分析
    """
    print("开始面板回归分析...")
    
    # 确保数据按股票代码和时间排序
    filtered_data = filtered_data.sort_values(by=['code', 'date'])

    # 对 AMBE 进行滞后一期处理
    filtered_data['AMBE_lag1'] = filtered_data.groupby('code')['AMBE'].shift(1)

    # 转换为面板格式（MultiIndex）
    filtered_data = filtered_data.set_index(['code', 'date'])

    # 删除缺失值
    filtered_data = filtered_data.dropna(subset=['AMBE_lag1', 'RV', 'Turnover_Rate'])

    # 定义变量（使用滞后变量）
    y = filtered_data['daily_log_return']
    X = filtered_data[['AMBE_lag1', 'RV', 'Turnover_Rate']]

    # 运行双向固定效应模型
    # 设定模型（个体+时间固定效应）
    model = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,  # 控制个体固定效应
        time_effects=True     # 控制时间固定效应
    )

    # 拟合模型（使用双向聚类标准误）
    results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

    # 提取回归结果的主要信息
    params = results.params  # 系数
    pvalues = results.pvalues  # P值
    conf_int = results.conf_int()  # 置信区间

    # 合并为 DataFrame
    results_df = pd.DataFrame({
        'Coefficient': params,
        'P-value': pvalues,
        'Lower CI': conf_int['lower'],
        'Upper CI': conf_int['upper']
    })

    print("面板回归分析完成")
    print(results_df)
    
    return results_df


def run_group_regression_by_ambe(filtered_data):
    """
    按AMBE分组进行回归分析
    """
    print("开始按AMBE分组回归分析...")
    
    # 确保数据按股票代码和时间排序
    filtered_data = filtered_data.sort_values(by=['code', 'date'])

    # 对 AMBE 进行滞后一期处理
    filtered_data['AMBE_lag1'] = filtered_data.groupby('code')['AMBE'].shift(1)

    # 转换为面板格式（MultiIndex）
    filtered_data = filtered_data.set_index(['code', 'date'])

    # 删除缺失值
    filtered_data = filtered_data.dropna(subset=['AMBE_lag1', 'RV', 'Turnover_Rate'])
    
    # 按 AMBE_lag1 分位数分组
    # 使用 pd.qcut 将 AMBE_lag1 分为三组
    filtered_data['AMBE_group'] = pd.qcut(filtered_data['AMBE_lag1'], q=3, labels=['Low', 'Medium', 'High'])

    # 初始化一个空的 DataFrame 用于存储结果
    results_df = pd.DataFrame()

    # 分组回归并保存结果
    for group in ['Low', 'Medium', 'High']:
        # 提取当前组的数据
        group_data = filtered_data[filtered_data['AMBE_group'] == group]
        
        # 定义因变量和自变量
        y = group_data['daily_log_return']
        X = group_data[['AMBE_lag1', 'RV', 'Turnover_Rate']]
        
        # 运行双向固定效应模型
        model = PanelOLS(
            dependent=y,
            exog=X,
            entity_effects=True,  # 控制个体固定效应
            time_effects=True     # 控制时间固定效应
        )
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        
        # 提取回归结果
        params = results.params  # 系数
        pvalues = results.pvalues  # P值
        conf_int = results.conf_int()  # 置信区间
        
        # 将结果保存为 DataFrame
        group_results = pd.DataFrame({
            'Variable': params.index,
            'Coefficient': params.values,
            'P-value': pvalues.values,
            'Lower CI': conf_int['lower'].values,
            'Upper CI': conf_int['upper'].values,
            'Group': group  # 添加分组标签
        })
        
        # 将当前组的结果追加到总结果中
        results_df = pd.concat([results_df, group_results], ignore_index=True)

    # 保存结果为 CSV 文件
    results_df.to_csv('分组回归结果.csv', index=False, encoding='utf_8_sig')
    print("分组回归结果已保存为 分组回归结果.csv")
    
    print("按AMBE分组回归分析完成")
    print(results_df)
    
    return results_df


def run_regression_with_controls(filtered_data):
    """
    运行包含控制变量的回归分析
    """
    print("开始带控制变量的回归分析...")
    
    # 存储每个显著的股票的回归结果
    significant_results = []
    significance_level = 0.05  # 设定显著性水平

    # 存储所有自变量的回归结果
    all_results = []

    # 对每个股票进行回归，使用 tqdm 添加进度条
    unique_codes = filtered_data['code'].unique()
    for code in tqdm(unique_codes, desc="Processing stocks", unit="stock"):
        # 提取当前股票的数据，并创建副本
        stock_data = filtered_data[filtered_data['code'] == code].copy()
        
        # 准备自变量和因变量，滞后自变量一期
        stock_data['AMBE_lagged'] = stock_data['AMBE'].shift(1)
        
        # 删除缺失值（由于滞后操作，第一行会有缺失值）
        stock_data = stock_data.dropna(subset=['AMBE_lagged', 'RV','Excess Return', 'Skewness', 'Kurtosis', 'Intraday_Range', 'Turnover_Rate', 'Intraday_Reversal'])
        
        # 更新自变量为滞后一期，并添加控制变量
        X = stock_data[['AMBE_lagged', 'RV','Skewness', 'Kurtosis', 'Intraday_Range', 'Turnover_Rate', 'Intraday_Reversal']]
        Y = stock_data['daily_log_return']
        
        # 添加常数项（截距）
        X = sm.add_constant(X)
        
        # 进行线性回归
        model = sm.OLS(Y, X).fit()
        
        # 提取 AMBE_lagged 的 p 值和系数
        p_value_ambe = model.pvalues['AMBE_lagged']
        coefficient_ambe = model.params['AMBE_lagged']
        
        # 检查 AMBE_lagged 是否显著
        if p_value_ambe < significance_level:
            # 存储结果
            significant_results.append({
                'code': code,
                'significance': p_value_ambe,
                'coefficient_magnitude': abs(coefficient_ambe),
                'coefficient_sign': 'positive' if coefficient_ambe > 0 else 'negative'
            })
        
        # 存储所有变量的回归结果
        results = {
            'code': code,
            'const_coef': model.params['const'],  # 常数项系数
            'const_p_value': model.pvalues['const'],  # 常数项 p 值
        }
        
        # 遍历每个变量，存储其系数、p 值、t 值等信息
        for var in X.columns:
            if var != 'const':  # 排除常数项
                results[f'{var}_coef'] = model.params[var]  # 系数
                results[f'{var}_p_value'] = model.pvalues[var]  # p 值
                results[f'{var}_t_value'] = model.tvalues[var]  # t 值
                results[f'{var}_significant'] = model.pvalues[var] < significance_level  # 是否显著
        
        # 将当前股票的结果添加到总结果中
        all_results.append(results)

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(significant_results)
    all_results_df = pd.DataFrame(all_results)

    # 保存结果为 CSV 文件
    results_df.to_csv('AMBE_显著回归结果.csv', index=False)
    all_results_df.to_csv('完整回归结果.csv', index=False)

    print("AMBE 显著回归结果已保存到 'AMBE_显著回归结果.csv'。")
    print("完整回归结果已保存到 '完整回归结果.csv'。")
    
    return results_df, all_results_df


def run_market_interaction_regression(filtered_data):
    """
    运行市场交互项回归分析
    """
    print("开始市场交互项回归分析...")
    
    # 存储每个显著的股票的回归结果
    significant_results = []
    significance_level = 0.05  # 设定显著性水平

    # 对每个股票进行回归，使用 tqdm 添加进度条
    unique_codes = filtered_data['code'].unique()
    for code in tqdm(unique_codes, desc="Processing stocks", unit="stock"):
        # 提取当前股票的数据，并创建副本
        stock_data = filtered_data[filtered_data['code'] == code].copy()
        
        # 对需要滞后的自变量进行滞后一期处理
        stock_data['AMBE_lagged'] = stock_data['AMBE'].shift(1)
        
        # 删除由于滞后操作产生的缺失值
        stock_data = stock_data.dropna(subset=[
            'AMBE_lagged', 'bull_market', 'bear_market', 'Excess Return', 
            'RV', 'Skewness', 'Kurtosis', 'Intraday_Range', 'Turnover_Rate', 'Intraday_Reversal'
        ])
        
        # 准备自变量和因变量
        # 使用滞后一期的 AMBE 计算交互项
        stock_data['BullMarket:ambiguity'] = stock_data['bull_market'] * stock_data['AMBE_lagged']
        stock_data['BearMarket:ambiguity'] = stock_data['bear_market'] * stock_data['AMBE_lagged']
        
        # 定义自变量（包括控制变量）
        X = stock_data[[
            'bull_market', 'bear_market', 'AMBE_lagged', 
            'BullMarket:ambiguity', 'BearMarket:ambiguity',
            'RV', 'Skewness', 'Kurtosis', 'Intraday_Range', 'Turnover_Rate', 'Intraday_Reversal'
        ]]
        Y = stock_data['daily_log_return']  # 使用 Excess Return 作为因变量
        
        # 添加常数项（截距）
        X = sm.add_constant(X)
        
        # 进行线性回归
        model = sm.OLS(Y, X).fit()
        
        # 提取滞后 AMBE 的 p 值和系数（如果需要）
        p_value_ambiguity = model.pvalues['AMBE_lagged']
        coefficient_ambiguity = model.params['AMBE_lagged']
        
        # 提取交互项的 p 值和系数
        p_value_bull_ambiguity = model.pvalues['BullMarket:ambiguity']
        coefficient_bull_ambiguity = model.params['BullMarket:ambiguity']
        
        p_value_bear_ambiguity = model.pvalues['BearMarket:ambiguity']
        coefficient_bear_ambiguity = model.params['BearMarket:ambiguity']
        
        # 检查交互项是否显著，并存储结果
        if p_value_bull_ambiguity < significance_level:
            significant_results.append({
                'code': code,
                'interaction': 'BullMarket:ambiguity',
                'p_value': p_value_bull_ambiguity,
                'coefficient_magnitude': abs(coefficient_bull_ambiguity),
                'coefficient_sign': 'positive' if coefficient_bull_ambiguity > 0 else 'negative'
            })
        
        if p_value_bear_ambiguity < significance_level:
            significant_results.append({
                'code': code,
                'interaction': 'BearMarket:ambiguity',
                'p_value': p_value_bear_ambiguity,
                'coefficient_magnitude': abs(coefficient_bear_ambiguity),
                'coefficient_sign': 'positive' if coefficient_bear_ambiguity > 0 else 'negative'
            })

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(significant_results)

    # 保存结果为 CSV 文件
    results_df.to_csv('分市场回归交互项.csv', index=False)

    print("Regression results for significant interaction terms have been saved to '分市场回归交互项.csv'.")
    
    return results_df


def analyze_by_rv_quantiles(filtered_data):
    """
    按RV分位数进行回归分析
    """
    print("开始按RV分位数进行回归分析...")
    
    # 计算RV的分位数
    q25 = filtered_data['RV'].quantile(0.33)
    q75 = filtered_data['RV'].quantile(0.66)

    # 根据RV的分位数将数据分为三组
    low_rv_data = filtered_data[filtered_data['RV'] <= q25]
    medium_rv_data = filtered_data[(filtered_data['RV'] > q25) & (filtered_data['RV'] <= q75)]
    high_rv_data = filtered_data[filtered_data['RV'] > q75]

    # 定义函数来运行回归
    def run_regression(data):
        y = data['daily_log_return']
        X = data[['AMBE_lag1', 'Turnover_Rate', 'Intraday_Range', 'Skewness', 'Kurtosis']]
        model = PanelOLS(dependent=y, exog=X, entity_effects=True, time_effects=True, drop_absorbed=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        return results

    # 分别运行低RV组、中RV组和高RV组的回归
    low_rv_results = run_regression(low_rv_data)
    medium_rv_results = run_regression(medium_rv_data)
    high_rv_results = run_regression(high_rv_data)

    # 定义函数来提取结果并保存为DataFrame
    def extract_results(results, group_name):
        params = results.params
        pvalues = results.pvalues
        conf_int = results.conf_int()
        results_df = pd.DataFrame({
            'Group': [group_name] * len(params),
            'Variable': params.index,
            'Coefficient': params.values,
            'P-value': pvalues.values,
            'Lower CI': conf_int['lower'].values,
            'Upper CI': conf_int['upper'].values
        })
        return results_df

    # 提取并保存三个组的结果
    low_rv_df = extract_results(low_rv_results, 'Low RV')
    medium_rv_df = extract_results(medium_rv_results, 'Medium RV')
    high_rv_df = extract_results(high_rv_results, 'High RV')

    # 合并三个DataFrame以便比较（可选）
    combined_results_df = pd.concat([low_rv_df, medium_rv_df, high_rv_df], ignore_index=True)

    print("按RV分位数回归分析完成")
    print("Low RV Group Results:")
    print(low_rv_df)
    print("\nMedium RV Group Results:")
    print(medium_rv_df)
    print("\nHigh RV Group Results:")
    print(high_rv_df)
    
    return low_rv_df, medium_rv_df, high_rv_df


def run_residual_regression(df):
    """
    运行残差回归分析
    """
    print("开始残差回归分析...")
    
    # Step 1: 生成滞后一期的 RV
    df['RV_lag1'] = df['RV'].shift(1)

    # Step 2: 回归 daily_log_return ~ RV_lag1，得到残差
    df_reg1 = df.dropna(subset=['daily_log_return', 'RV_lag1'])  # 去除NaN行
    X1 = sm.add_constant(df_reg1['RV_lag1'])
    y1 = df_reg1['daily_log_return']
    model1 = sm.OLS(y1, X1).fit()
    df_reg1['residual'] = model1.resid  # 残差作为新列

    # Step 3: 用残差回归 AMBE
    X2 = sm.add_constant(df_reg1['AMBE'])
    y2 = df_reg1['residual']
    model2 = sm.OLS(y2, X2).fit()

    print("\nStep 2 回归: residual ~ AMBE")
    print(model2.summary())

    # 可选：更新回原 df（如果你想保留 residual 这一列）
    df.loc[df_reg1.index, 'residual'] = df_reg1['residual']
    
    return df, model2


def run_residual_regression_analysis(filtered_data):
    """
    运行信念风险和不确定性分析
    """
    print("开始信念风险和不确定性分析...")
    
    # 对 AMBE 进行滞后一期处理
    filtered_data['AMBE_lag1'] = filtered_data.groupby('code')['AMBE'].shift(1)
    # 对换手率和RV进行滞后一期处理
    filtered_data['Turnover_Rate_lag1'] = filtered_data.groupby('code')['Turnover_Rate'].shift(1)  # 换手率滞后一期
    filtered_data['RV_lag1'] = filtered_data.groupby('code')['RV'].shift(1)  # RV滞后一期

    # 选择需要归一化的列
    columns_to_normalize = ['AMBE_lag1', 'Turnover_Rate_lag1', 'RV_lag1']
    
    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()
    
    # 按 'code' 分组并应用归一化函数
    def normalize_group(group):
        # 对每组数据应用 MinMaxScaler
        if len(group) > 1:  # 确保组内有足够数据进行归一化
            group[columns_to_normalize] = scaler.fit_transform(group[columns_to_normalize])
        return group
    
    # 按 'code' 分组并应用归一化函数
    filtered_data = filtered_data.groupby('code').apply(normalize_group)

    # 对AMBE_lag1进行1%缩尾处理
    lower_bound = filtered_data['AMBE_lag1'].quantile(0.01)
    upper_bound = filtered_data['AMBE_lag1'].quantile(0.99)
    filtered_data['AMBE_lag1'] = filtered_data['AMBE_lag1'].clip(lower_bound, upper_bound)
    
    # 确保数据按股票代码和时间排序
    filtered_data = filtered_data.sort_values(by=['code', 'date'])

    # 删除缺失值（由于滞后操作会产生NA）
    filtered_data = filtered_data.dropna(subset=['Turnover_Rate_lag1', 'RV_lag1','AMBE_lag1'])

    # 因变量：超额收益（确保已计算）
    y = filtered_data['daily_log_return']*100 # 假设 daily_log_return 是超额收益

    # 自变量与控制变量
    exog_vars = [
        'AMBE_lag1',             # 模糊度（滞后一期）
        'RV_lag1',               # 风险（已实现波动率，滞后一期）
        'Turnover_Rate_lag1',    # 信念（换手率，滞后一期）
    ]
    X = filtered_data[exog_vars]

    # 运行双向固定效应模型
    # 设定模型（控制个体和时间固定效应）
    model = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,  # 控制个体固定效应
        time_effects=True,    # 控制时间固定效应
        drop_absorbed=True    # 自动删除被吸收的变量
    )

    # 拟合模型（使用双向聚类标准误）
    results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

    # 提取关键参数
    params = results.params
    pvalues = results.pvalues
    conf_int = results.conf_int()

    # 将结果保存为 DataFrame
    results_df = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'P-value': pvalues.values,
        'Lower CI': conf_int['lower'].values,
        'Upper CI': conf_int['upper'].values
    })

    # 打印结果表格
    print("信念风险和不确定性分析完成")
    print(results_df)
    
    return results_df


def run_ambe_by_sample_analysis(filtered_data):
    """
    不同模糊度样本下，信念效应的影响分析
    """
    print("开始不同模糊度样本下信念效应影响分析...")
    
    # 按AMBE水平分组（前30%为高模糊性，后70%为低模糊性）
    high_ambe_threshold = filtered_data['AMBE_lag1'].quantile(0.8)
    high_ambe_data = filtered_data[filtered_data['AMBE_lag1'] > high_ambe_threshold]  # 高模糊性子样本
    low_ambe_data = filtered_data[filtered_data['AMBE_lag1'] <= high_ambe_threshold]  # 低模糊性子样本

    # 定义回归函数
    def run_regression(data):
        # 因变量：超额收益
        y = data['daily_log_return']*100
        
        # 自变量与控制变量
        exog_vars = [
            'AMBE_lag1',             # 模糊度（滞后一期）
            'RV_lag1',               # 风险（已实现波动率，滞后一期）
            'Turnover_Rate_lag1',    # 信念（换手率，滞后一期）
        ]
        X = data[exog_vars]
        
        # 设定模型（控制个体和时间固定效应）
        model = PanelOLS(
            dependent=y,
            exog=X,
            entity_effects=True,  # 控制个体固定效应
            time_effects=True,    # 控制时间固定效应
            drop_absorbed=True    # 自动删除被吸收的变量
        )
        
        # 拟合模型（使用双向聚类标准误）
        results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        return results

    # 高模糊性子样本回归
    high_ambe_results = run_regression(high_ambe_data)

    # 低模糊性子样本回归
    low_ambe_results = run_regression(low_ambe_data)

    # 提取结果并对比
    def extract_results(results, group_name):
        params = results.params
        pvalues = results.pvalues
        conf_int = results.conf_int()
        
        # 将结果保存为 DataFrame
        results_df = pd.DataFrame({
            'Group': group_name,
            'Variable': params.index,
            'Coefficient': params.values,
            'P-value': pvalues.values,
            'Lower CI': conf_int['lower'].values,
            'Upper CI': conf_int['upper'].values
        })
        return results_df

    # 提取高模糊性子样本结果
    high_ambe_df = extract_results(high_ambe_results, 'High AMBE')

    # 提取低模糊性子样本结果
    low_ambe_df = extract_results(low_ambe_results, 'Low AMBE')

    # 合并结果
    combined_results_df = pd.concat([high_ambe_df, low_ambe_df], ignore_index=True)

    # 输出结果
    print("高模糊性子样本结果:")
    print(high_ambe_df)
    print("\n低模糊性子样本结果:")
    print(low_ambe_df)
    
    return high_ambe_df, low_ambe_df, combined_results_df


def main_analysis():
    """
    执行完整的回归分析流程
    """
    print("开始完整的回归分析流程...")
    
    # 1. 加载和合并数据
    data = load_and_merge_data()
    print(f"加载的数据形状: {data.shape}")
    
    # 2. 过滤股票
    filtered_data = filter_stocks_by_date(data)
    print(f"过滤后的数据形状: {filtered_data.shape}")
    
    # 3. 计算沪深300指数收益率
    result_df = calculate_index_return()
    
    # 4. 计算超额收益
    filtered_data = calculate_excess_return(filtered_data, result_df)
    print(f"计算超额收益后的数据形状: {filtered_data.shape}")
    
    # 5. 保存预处理后的数据
    filtered_data.to_parquet("回归数据.pqt")
    filtered_data.to_parquet("模糊度回归分析原始数据", engine='pyarrow')
    
    # 6. 分析牛熊市条件
    filtered_data = analyze_market_conditions(filtered_data)
    
    # 7. 运行面板回归
    panel_results = run_panel_regression(filtered_data.copy())
    
    # 8. 按AMBE分组回归
    group_results = run_group_regression_by_ambe(filtered_data.copy())
    
    # 9. 运行带控制变量的回归
    significant_results, all_results = run_regression_with_controls(filtered_data)
    
    # 10. 运行市场交互项回归
    market_interaction_results = run_market_interaction_regression(filtered_data)
    
    # 11. 按RV分位数进行回归分析
    low_rv_df, medium_rv_df, high_rv_df = analyze_by_rv_quantiles(filtered_data)
    
    # 12. 基于牛熊市的面板回归
    market_based_results = run_market_based_panel_regression(filtered_data.copy())
    
    # 13. 按价格水平分组回归
    price_level_results = analyze_by_price_levels(filtered_data.copy())
    
    # 13.5 按不同价位进行回归分析
    low_plevel_df, medium_plevel_df, high_plevel_df = analyze_by_price_levels_regression(filtered_data.copy())
    
    # 14. 残差回归分析
    df = pd.read_parquet("模糊度回归分析原始数据", engine='pyarrow')
    residual_results, model = run_residual_regression(df)
    
    # 15. 滚动回归分析
    rolling_results = analyze_rolling_regression(filtered_data.copy())
    
    # 16. 信念风险和不确定性分析
    belief_risk_results = run_residual_regression_analysis(filtered_data.copy())
    
    # 17. 不同模糊度样本下信念效应的影响分析
    high_ambe_df, low_ambe_df, combined_ambe_results = run_ambe_by_sample_analysis(filtered_data.copy())
    
    print("完整回归分析流程完成！")
    
    # 返回主要结果
    return {
        'panel_results': panel_results,
        'group_results': group_results,
        'significant_results': significant_results,
        'all_results': all_results,
        'market_interaction_results': market_interaction_results,
        'rv_quantile_results': (low_rv_df, medium_rv_df, high_rv_df),
        'market_based_results': market_based_results,
        'price_level_results': price_level_results,
        'plevel_regression_results': (low_plevel_df, medium_plevel_df, high_plevel_df),
        'residual_results': residual_results,
        'rolling_results': rolling_results,
        'belief_risk_results': belief_risk_results,
        'ambe_sample_results': (high_ambe_df, low_ambe_df, combined_ambe_results)
    }


if __name__ == "__main__":
    # 执行主分析流程
    results = main_analysis()