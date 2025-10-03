import pandas as pd
import os
import numpy as np

# 设置文件路径
climate_file = '/Users/tlxy/Research/Ambiguity/data/Climate_Risk_Index.xlsx'
gpr_file = '/Users/tlxy/Research/Ambiguity/data/data_gpr_export.xls'
output_dir = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print("开始提取气候风险数据...")
# 直接读取气候风险数据，跳过前6行说明内容
df_climate = pd.read_excel(climate_file, sheet_name='Climate Risk Data Rognone', skiprows=6)

# 重命名列以使其更清晰
columns = ['Date'] + [f'Climate_Risk_{i}' for i in range(1, 15)]
df_climate.columns = columns

# 过滤掉非日期行和空行
df_climate_clean = df_climate[pd.to_datetime(df_climate['Date'], errors='coerce').notna()].copy()

# 确保日期列格式正确
df_climate_clean['Date'] = pd.to_datetime(df_climate_clean['Date'])

# 清理数据：将非数值列转换为数值，无法转换的设为NaN
df_climate_numeric = df_climate_clean.copy()
for col in df_climate_numeric.columns[1:]:  # 从第二列开始处理风险指标
    # 尝试转换为数值
    df_climate_numeric[col] = pd.to_numeric(df_climate_numeric[col], errors='coerce')

# 过滤掉所有风险指标都是NaN的行
df_climate_final = df_climate_numeric.dropna(subset=df_climate_numeric.columns[1:], how='all')

# 只保留日期和有数值的风险指标列
df_climate_clean = df_climate_final

# 保存气候风险数据到CSV
climate_output_file = os.path.join(output_dir, 'climate_risk_series_daily.csv')
df_climate_clean.to_csv(climate_output_file, index=False)
print(f"气候风险数据已保存到: {climate_output_file}")
print(f"数据形状: {df_climate_clean.shape}")
print(f"数据列: {df_climate_clean.columns.tolist()}")
print(f"数据时间范围: {df_climate_clean.iloc[0, 0]} 到 {df_climate_clean.iloc[-1, 0]}")

print("\n开始提取GPR国家数据...")
# 提取GPR数据，只包含中国、香港、日本和美国的数据
df_gpr = pd.read_excel(gpr_file)

# 选择需要的列：日期和目标国家数据
country_columns = {
    'month': 'Date',
    'GPRHC_CHN': 'China',
    'GPRHC_HKG': 'Hongkong', 
    'GPRHC_JPN': 'Japan',
    'GPRHC_USA': 'US'
}

# 筛选数据，排除前几行的说明行
# 找到实际数据开始的行（所有主要列都有数值的行）
valid_rows = df_gpr[['GPRHC_CHN', 'GPRHC_HKG', 'GPRHC_JPN', 'GPRHC_USA']].notna().all(axis=1)
df_gpr_clean = df_gpr[valid_rows].copy()

# 只保留需要的列并重命名
df_gpr_countries = df_gpr_clean[list(country_columns.keys())].rename(columns=country_columns)

# 保存GPR国家数据到CSV
gpr_output_file = os.path.join(output_dir, 'gpr_countries_data.csv')
df_gpr_countries.to_csv(gpr_output_file, index=False)
print(f"GPR国家数据已保存到: {gpr_output_file}")
print(f"数据形状: {df_gpr_countries.shape}")
print(f"数据列: {df_gpr_countries.columns.tolist()}")
print(f"数据时间范围: {df_gpr_countries.iloc[0, 0]} 到 {df_gpr_countries.iloc[-1, 0]}")

print("\n数据提取和保存完成！")