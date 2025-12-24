# -*- coding: utf-8 -*-
"""
数据转换

该脚本实现了将.mat文件转换为CSV文件的功能，包括：
1. 从.mat文件中提取二维数组数据
2. 将数据转换为DataFrame格式
3. 删除不需要的列
4. 保存为CSV文件
"""

import os
from scipy.io import loadmat
import pandas as pd
import numpy as np


def convert_mat_to_csv(mat_folder_path='gtaMin_1_new', csv_folder_path='全部股票数据'):
    """
    将指定文件夹中的所有.mat文件转换为CSV文件
    
    Parameters:
    mat_folder_path (str): .mat文件所在的文件夹路径
    csv_folder_path (str): 输出CSV文件的文件夹路径
    """
    print("开始.mat到CSV转换...")
    
    # 数据文件列名（假设这是所有可能列名的完整列表）
    column_names = ['code', 'date', 'a', 'open', 'high', 'low', 'close', 'value', 'amount', 'Turnover rate',
                    'Csv circulated stock value', 'capitalization', 'floating shares', 'total number', 'Adjusted factor',
                    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

    # 要删除的列名（可以根据实际情况调整）
    columns_to_drop = ['Turnover rate', 'Csv circulated stock value', 'capitalization', 'floating shares', 'total number',
                       'Adjusted factor'] + ['{}'.format(i) for i in range(1, 19)]  # 添加数字列名1到18

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # 获取文件夹中的所有.mat文件
    selected_files = [f for f in os.listdir(mat_folder_path) if f.endswith('.mat')]

    # 初始化错误计数器
    error_count = 0

    # 遍历每个文件
    for filename in selected_files:
        file_path = os.path.join(mat_folder_path, filename)
        file_report = []  # 用于存储当前文件的报告信息

        try:
            # 加载.mat文件
            data = loadmat(file_path)

            # 遍历.mat文件中的所有键
            for key, value in data.items():
                # 检查值是否是二维numpy数组
                if isinstance(value, np.ndarray) and len(value.shape) == 2:
                    # 根据数组的列数截取列名列表
                    num_cols = value.shape[1]
                    used_column_names = column_names[:num_cols]

                    # 创建DataFrame
                    df = pd.DataFrame(value, columns=used_column_names)

                    # 删除指定的列
                    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

                    # 定义CSV文件的路径和名称
                    csv_file_name = os.path.splitext(filename)[0] + '.csv'
                    csv_file_path = os.path.join(csv_folder_path, csv_file_name)

                    # 保存DataFrame为CSV文件
                    df.to_csv(csv_file_path, index=False)

                    # 添加当前文件的报告信息
                    file_report.append(f"文件 {filename} 处理成功，并保存为 {csv_file_name}")
                    # 如果.mat文件中只包含一个我们关心的二维数组，可以在这里跳出循环
                    # 但如果可能有多个，就不要跳出
                    break  # 假设每个.mat文件只包含一个我们需要的二维数组，所以跳出循环

        except Exception as e:
            error_count += 1
            file_report.append(f"Error loading {file_path}: {e}")

        # 输出当前文件的报告
        print("\n".join(file_report))
        print("-" * 40)  # 分隔符，用于区分不同文件的报告

    # 打印最终处理报告
    print("\n最终处理报告:")
    print(f"总共处理了 {len(selected_files)} 个文件。")
    print(f"遇到了 {error_count} 个错误。")
    if error_count == 0:
        print("所有文件都成功处理并保存为CSV！")
    else:
        print(f"有 {error_count} 个文件处理失败。")


def main():
    """
    主函数，执行完整的数据转换流程
    """
    print("开始执行数据转换流程...")
    
    # 执行.mat到CSV转换
    convert_mat_to_csv()
    
    print("数据转换流程完成！")


if __name__ == "__main__":
    main()