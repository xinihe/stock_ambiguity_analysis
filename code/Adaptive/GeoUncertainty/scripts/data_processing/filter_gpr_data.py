import pandas as pd
import os

# 设置文件路径
gpr_file = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/gpr_countries_data.csv'
output_dir = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty'

print("开始过滤GPR数据...")

# 读取GPR数据
df_gpr = pd.read_csv(gpr_file)
print(f"原始数据形状: {df_gpr.shape}")
print(f"原始日期范围: {df_gpr['Date'].min()} 到 {df_gpr['Date'].max()}")

# 确保日期格式正确
df_gpr['Date'] = pd.to_datetime(df_gpr['Date'])

# 过滤掉2005年1月1日之前的数据
df_gpr_filtered = df_gpr[df_gpr['Date'] >= '2005-01-01'].copy()

print(f"\n过滤后的数据形状: {df_gpr_filtered.shape}")
print(f"过滤后日期范围: {df_gpr_filtered['Date'].min().strftime('%Y-%m-%d')} 到 {df_gpr_filtered['Date'].max().strftime('%Y-%m-%d')}")
print(f"共过滤掉 {len(df_gpr) - len(df_gpr_filtered)} 行数据")

# 保存过滤后的数据
filtered_gpr_file = os.path.join(output_dir, 'gpr_countries_data_filtered.csv')
df_gpr_filtered.to_csv(filtered_gpr_file, index=False)

print(f"\n过滤后的GPR数据已保存到: {filtered_gpr_file}")
print("数据过滤完成！")