import pandas as pd
import os
import numpy as np

# 设置文件路径
climate_file = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/climate_risk_series_daily.csv'
gpr_file = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/gpr_countries_data.csv'
output_dir = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty'

print("开始清理气候风险数据...")
# 读取气候风险数据
df_climate = pd.read_csv(climate_file)

# 打印原始信息
print(f"原始数据形状: {df_climate.shape}")
print(f"列缺失值统计:\n{df_climate.isna().sum()}")

# 删除完全为空的列
df_climate_clean = df_climate.dropna(axis=1, how='all')
print(f"\n删除空列后的数据形状: {df_climate_clean.shape}")

# 为列设置更可读的名称
column_mapping = {
    'Date': 'Date',
    'Climate_Risk_1': 'Physical_Risk_Index',
    'Climate_Risk_2': 'Physical_Risk_Change',
    'Climate_Risk_3': 'Transition_Risk_Index',
    'Climate_Risk_4': 'Transition_Risk_Change',
    'Climate_Risk_6': 'Climate_Policy_Risk',
    'Climate_Risk_9': 'Market_Sentiment_Risk'
}

df_climate_clean = df_climate_clean.rename(columns=column_mapping)
print(f"\n新的列名: {df_climate_clean.columns.tolist()}")

# 确保日期格式正确
df_climate_clean['Date'] = pd.to_datetime(df_climate_clean['Date'])

# 过滤掉早于2005-01-01的数据（如果有的话）
df_climate_final = df_climate_clean[df_climate_clean['Date'] >= '2005-01-01'].copy()
print(f"\n过滤后的数据形状: {df_climate_final.shape}")
print(f"日期范围: {df_climate_final['Date'].min()} 到 {df_climate_final['Date'].max()}")

# 保存清理后的数据
clean_climate_file = os.path.join(output_dir, 'climate_risk_series_daily_clean.csv')
df_climate_final.to_csv(clean_climate_file, index=False)
print(f"\n清理后的气候风险数据已保存到: {clean_climate_file}")

# 读取GPR国家数据
df_gpr = pd.read_csv(gpr_file)
df_gpr['Date'] = pd.to_datetime(df_gpr['Date'])
print(f"\nGPR数据形状: {df_gpr.shape}")
print(f"GPR数据日期范围: {df_gpr['Date'].min()} 到 {df_gpr['Date'].max()}")

# 创建数据信息文档
data_info_content = f"""# 数据信息文档

## 1. 气候风险数据 (climate_risk_series_daily_clean.csv)

### 数据来源
原始数据来源于 Climate_Risk_Index.xlsx 文件中的气候风险序列数据。

### 数据处理步骤
1. 从Excel文件中提取气候风险序列的日数据
2. 跳过文件前6行的说明内容
3. 将列重命名为更可读的名称
4. 删除完全为空的列
5. 确保日期格式正确
6. 过滤掉早于2005-01-01的数据（虽然实际数据从2005-01-03开始）
7. 保存为CSV格式

### 数据概览
- **数据行数**: {df_climate_final.shape[0]}
- **数据列数**: {df_climate_final.shape[1]}
- **日期范围**: {df_climate_final['Date'].min().strftime('%Y-%m-%d')} 到 {df_climate_final['Date'].max().strftime('%Y-%m-%d')}

### 数据列说明
"""

# 添加气候风险数据列说明
climate_columns_info = {
    'Date': '日期（YYYY-MM-DD格式）',
    'Physical_Risk_Index': '物理风险指数',
    'Physical_Risk_Change': '物理风险变化率',
    'Transition_Risk_Index': '转型风险指数',
    'Transition_Risk_Change': '转型风险变化率',
    'Climate_Policy_Risk': '气候政策风险指标',
    'Market_Sentiment_Risk': '市场情绪风险指标'
}

for col, desc in climate_columns_info.items():
    if col in df_climate_final.columns:
        data_info_content += f"- **{col}**: {desc}\n"

data_info_content += f"""

## 2. GPR国家数据 (gpr_countries_data.csv)

### 数据来源
原始数据来源于 data_gpr_export.xls 文件中的地缘政治风险数据。

### 数据处理步骤
1. 从Excel文件中提取地缘政治风险数据
2. 筛选出中国、香港、日本和美国四个国家/地区的数据
3. 重命名列以使其更清晰
4. 排除说明行，只保留实际数据
5. 确保日期格式正确
6. 保存为CSV格式

### 数据概览
- **数据行数**: {df_gpr.shape[0]}
- **数据列数**: {df_gpr.shape[1]}
- **日期范围**: {df_gpr['Date'].min().strftime('%Y-%m-%d')} 到 {df_gpr['Date'].max().strftime('%Y-%m-%d')}

### 数据列说明
"""

# 添加GPR数据列说明
gpr_columns_info = {
    'Date': '日期（YYYY-MM-DD格式）',
    'China': '中国地缘政治风险指数',
    'Hongkong': '香港地缘政治风险指数',
    'Japan': '日本地缘政治风险指数',
    'US': '美国地缘政治风险指数'
}

for col, desc in gpr_columns_info.items():
    if col in df_gpr.columns:
        data_info_content += f"- **{col}**: {desc}\n"

# 保存数据信息文档
data_info_file = os.path.join(output_dir, 'data_info.md')
with open(data_info_file, 'w', encoding='utf-8') as f:
    f.write(data_info_content)

print(f"\n数据信息文档已保存到: {data_info_file}")
print("\n数据清理和文档生成完成！")