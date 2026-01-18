"""
金融因子分析与回归模型
包含IC计算、相关性分析、基准回归、分组回归和回测分析
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from linearmodels.panel import PanelOLS
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt


def load_data(file_path="回归数据.pqt"):
    """
    加载回归分析数据
    """
    df = pd.read_parquet(file_path, engine='pyarrow')
    return df


def calculate_correlations(df, target_col='AMBE'):
    """
    计算AMBE或AMBMP与RV、偏度、峰度的相关性
    """
    correlation_results = pd.DataFrame(
        columns=['Correlation', 'p-value'], 
        index=['RV', 'Skewness', 'Kurtosis']
    )

    for col in ['RV', 'Skewness', 'Kurtosis']:
        corr, p_value = pearsonr(df[target_col], df[col])
        correlation_results.loc[col] = [corr, p_value]

    return correlation_results


def calculate_vif(df):
    """
    计算方差膨胀因子(VIF)，检查多重共线性
    """
    X = df[['AMBMP', 'AMBE', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate']]  
    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data


def prepare_market_data(df):
    """
    准备市场状态数据（牛市/熊市虚拟变量）
    """
    # 牛市和熊市时间段
    bull_markets = [
        ('1990-12-19', '1992-05-26'),
        ('1992-11-17', '1993-02-16'),
        ('1994-07-29', '1994-09-13'),
        ('1996-01-19', '1997-05-12'),
        ('1999-05-19', '2001-06-14'),
        ('2005-06-06', '2007-10-16'),
        ('2008-10-28', '2009-08-04'),
        ('2013-06-25', '2015-06-12'),
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

    # 将日期字符串转化为 datetime 对象
    bull_markets = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bull_markets]
    bear_markets = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in bear_markets]

    # 创建牛市和熊市的虚拟变量
    def is_bull_market(date):
        for start, end in bull_markets:
            if start <= date <= end:
                return 1
        return 0

    def is_bear_market(date):
        for start, end in bear_markets:
            if start <= date <= end:
                return 1
        return 0

    # 假设 df 中有 'date' 列，且已经是 datetime 格式
    df['bull_market'] = df['date'].apply(is_bull_market)
    df['bear_market'] = df['date'].apply(is_bear_market)

    return df


def prepare_panel_data(df):
    """
    准备面板数据，包括标准化和创建因变量
    """
    # 重置索引，以便设置面板索引
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # 创建 MultiIndex：面板结构 = (code, date)
    df = df.set_index(['code', 'date'])

    # 计算下一期收益作为因变量
    df['next_return'] = df.groupby(level=0)['daily_log_return'].shift(-1)

    # 选取自变量和因变量，并去除缺失值
    variables = ['next_return', 'AMBE', 'AMBMP', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate', 'bull_market', 'bear_market','plevel']
    panel_df = df[variables].dropna()

    # 按 'code' 分组标准化所有变量（包括因变量）
    panel_df[variables] = (
        panel_df
        .groupby(level=0)[variables]
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    )

    return panel_df, variables


def calculate_vif_after_standardization(panel_df, variables):
    """
    计算标准化后各变量的VIF
    """
    X_std = add_constant(panel_df[['AMBE', 'AMBMP', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate']])
    df_vif = pd.DataFrame({
        'Variable': X_std.columns,
        'VIF': [variance_inflation_factor(X_std.values, i) for i in range(X_std.shape[1])]
    })
    return df_vif


def run_baseline_regression(panel_df):
    """
    运行基准回归分析
    """
    # 回归 1：next_return ~ AMBE + 控制变量 + 固定效应
    model_ambe = PanelOLS.from_formula(
        'next_return ~ AMBE + RV + Skewness + Kurtosis + Turnover_Rate + EntityEffects',
        data=panel_df
    ).fit(cov_type='clustered', cluster_entity=True)

    # 回归 2：next_return ~ AMBMP + 控制变量 + 固定效应
    model_ambmp = PanelOLS.from_formula(
        'next_return ~ AMBMP + RV + Skewness + Kurtosis + Turnover_Rate + EntityEffects',
        data=panel_df
    ).fit(cov_type='clustered', cluster_entity=True)

    return model_ambe, model_ambmp


def run_market_regression(panel_df):
    """
    运行分市场回归分析
    """
    # AMBE 模型回归
    model_ambe = PanelOLS.from_formula(
        'next_return ~ AMBE + RV + Skewness + Kurtosis + Turnover_Rate + bull_market + bear_market + EntityEffects',
        data=panel_df
    ).fit(cov_type='clustered', cluster_entity=True)

    # AMBMP 模型回归
    model_ambmp = PanelOLS.from_formula(
        'next_return ~ AMBMP + RV + Skewness + Kurtosis + Turnover_Rate + bull_market + bear_market + EntityEffects',
        data=panel_df
    ).fit(cov_type='clustered', cluster_entity=True)

    return model_ambe, model_ambmp


def run_rv_group_regression(panel_df):
    """
    根据RV分组进行回归分析
    """
    # 计算RV的33分位数
    rv_33th = panel_df['RV'].quantile(0.33)

    # 分成低RV组和高RV组
    low_rv_group = panel_df[panel_df['RV'] <= rv_33th]
    high_rv_group = panel_df[panel_df['RV'] > rv_33th]

    # 定义回归函数
    def run_panel_regression(group, dependent_var, independent_vars):
        # 使用面板数据回归，设置EntityEffects（固定效应）
        model = PanelOLS.from_formula(
            f'{dependent_var} ~ ' + ' + '.join(independent_vars) + ' + EntityEffects',
            data=group
        ).fit(cov_type='clustered', cluster_entity=True)
        return model

    # 低RV组回归（AMBE 和 AMBMP）
    low_rv_ambe_model = run_panel_regression(low_rv_group, 'next_return', ['AMBE', 'Skewness', 'Kurtosis', 'Turnover_Rate'])
    low_rv_ambmp_model = run_panel_regression(low_rv_group, 'next_return', ['AMBMP', 'Skewness', 'Kurtosis', 'Turnover_Rate'])

    # 高RV组回归（AMBE 和 AMBMP）
    high_rv_ambe_model = run_panel_regression(high_rv_group, 'next_return', ['AMBE', 'Skewness', 'Kurtosis', 'Turnover_Rate'])
    high_rv_ambmp_model = run_panel_regression(high_rv_group, 'next_return', ['AMBMP', 'Skewness', 'Kurtosis', 'Turnover_Rate'])

    return low_rv_ambe_model, low_rv_ambmp_model, high_rv_ambe_model, high_rv_ambmp_model


def run_plevel_group_regression(panel_df):
    """
    根据plevel分组进行回归分析
    """
    # 分位数阈值
    low_threshold = panel_df['plevel'].quantile(0.33)
    high_threshold = panel_df['plevel'].quantile(0.66)

    # 分组
    low_plevel = panel_df[panel_df['plevel'] <= low_threshold]
    high_plevel = panel_df[panel_df['plevel'] >= high_threshold]

    # 回归模型
    model_low_ambe = PanelOLS.from_formula('next_return ~ AMBE + RV+Skewness + Kurtosis + Turnover_Rate + EntityEffects',
                                       data=low_plevel).fit(cov_type='clustered', cluster_entity=True)
    model_low_ambmp = PanelOLS.from_formula('next_return ~ AMBMP + RV+Skewness + Kurtosis + Turnover_Rate + EntityEffects',
                                        data=low_plevel).fit(cov_type='clustered', cluster_entity=True)

    model_high_ambe = PanelOLS.from_formula('next_return ~ AMBE + RV+Skewness + Kurtosis + Turnover_Rate + EntityEffects',
                                        data=high_plevel).fit(cov_type='clustered', cluster_entity=True)
    model_high_ambmp = PanelOLS.from_formula('next_return ~ AMBMP + RV+Skewness + Kurtosis + Turnover_Rate + EntityEffects',
                                         data=high_plevel).fit(cov_type='clustered', cluster_entity=True)

    return model_low_ambe, model_low_ambmp, model_high_ambe, model_high_ambmp


def run_quantile_regression(df):
    """
    进行五组分组回归分析
    """
    # 确保 'code' 和 'date' 是列
    df = df.reset_index()

    # 创建 MultiIndex：面板结构 = (code, date)
    df = df.set_index(['code', 'date'])

    # 构造因变量和自变量（并去除缺失值）
    df['next_return'] = df.groupby(level=0)['daily_log_return'].shift(-1)
    panel_df = df[['next_return', 'AMBE', 'AMBMP', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate']].dropna()

    # 变量标准化（包括因变量和所有自变量），按 code 分组
    for col in ['next_return', 'AMBE', 'AMBMP', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate']:
        panel_df[col] = panel_df.groupby('code')[col].transform(lambda x: (x - x.mean()) / x.std())

    # 重置索引，确保有 'code' 列可用于分组
    panel_df = panel_df.reset_index()

    # 分组变量：基于标准化后的 AMBE 分为五组
    panel_df['AMBE_group'] = pd.qcut(panel_df['AMBE'], 5, labels=False)

    # 回归结果保存字典
    regression_results = {'AMBE': [], 'AMBMP': []}

    # 遍历每个 AMBE 分组
    for group in range(5):
        group_data = panel_df[panel_df['AMBE_group'] == group]
        group_data = group_data.set_index(['code', 'date'])  # 重新设回 MultiIndex
        
        # 回归 1：AMBE + 控制变量 + 固定效应
        model_ambe = PanelOLS.from_formula(
            'next_return ~ AMBE + RV + Skewness + Kurtosis + Turnover_Rate + EntityEffects',
            data=group_data
        ).fit(cov_type='clustered', cluster_entity=True)
        
        regression_results['AMBE'].append({
            'Coefficient': model_ambe.params['AMBE'],
            'Std. Error': model_ambe.std_errors['AMBE'],
            't-stat': model_ambe.tstats['AMBE'],
            'p-value': model_ambe.pvalues['AMBE'],
            'R-squared': model_ambe.rsquared
        })
        
        # 回归 2：AMBMP + 控制变量 + 固定效应
        model_ambmp = PanelOLS.from_formula(
            'next_return ~ AMBMP + RV + Skewness + Kurtosis + Turnover_Rate + EntityEffects',
            data=group_data
        ).fit(cov_type='clustered', cluster_entity=True)
        
        regression_results['AMBMP'].append({
            'Coefficient': model_ambmp.params['AMBMP'],
            'Std. Error': model_ambmp.std_errors['AMBMP'],
            't-stat': model_ambmp.tstats['AMBMP'],
            'p-value': model_ambmp.pvalues['AMBMP'],
            'R-squared': model_ambmp.rsquared
        })

    # 输出 DataFrame
    ambe_df = pd.DataFrame(regression_results['AMBE'])
    ambmp_df = pd.DataFrame(regression_results['AMBMP'])

    return ambe_df, ambmp_df


def run_granger_causality_test(df):
    """
    进行Granger因果检验
    """
    # 准备数据（选择适当的列进行因果检验）
    granger_data = df[['next_return', 'AMBMP']].dropna()

    # 执行Granger因果检验，最大滞后期数设为5
    gc_result = grangercausalitytests(granger_data, maxlag=5, verbose=True)

    # 提取检验结果并整理成一个简洁的表格
    granger_summary = {}

    for lag in range(1, 6):  # 1到5次滞后期
        f_stat = gc_result[lag][0]['ssr_ftest'][0]
        p_value = gc_result[lag][0]['ssr_ftest'][1]
        granger_summary[lag] = {'F-statistic': f_stat, 'p-value': p_value}

    # 将结果存储为DataFrame
    granger_df = pd.DataFrame(granger_summary).T

    return granger_df


def load_backtest_data(file_path="模糊度回归分析原始数据.pqt"):
    """
    加载回测数据
    """
    data = pd.read_parquet(file_path, engine='pyarrow')
    return data


def load_hs300_data(file_path='沪深300日收益率.csv'):
    """
    加载沪深300数据
    """
    hs300 = pd.read_csv(file_path, parse_dates=['Date'])
    hs300.columns = ['date', 'hs300_return']
    hs300.set_index('date', inplace=True)
    return hs300


def calc_rank_ic(df, factor):
    """
    计算RankIC和RankICIR
    """
    rank_ic = df.groupby('date').apply(
        lambda x: spearmanr(x[factor], x['daily_log_return'])[0]
    )
    rank_ic.name = 'RankIC'
    rank_icir = rank_ic.mean() / rank_ic.std() * np.sqrt(252)
    return rank_ic, rank_icir


def calc_annual_return_max_drawdown(group_return):
    """
    计算年化收益率和最大回撤
    """
    cum_ret = (1 + group_return).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    ann_ret = cum_ret.iloc[-1] ** (252 / len(group_return)) - 1
    max_dd = drawdown.min()
    return ann_ret, max_dd


def factor_group_returns(df, factor, n_quantiles=10):
    """
    计算因子分组收益率
    """
    df = df.copy()
    results = []

    for date, group in df.groupby('date'):
        group = group.dropna(subset=[factor, 'daily_log_return'])
        if len(group) < n_quantiles: continue
        group['quantile'] = pd.qcut(group[factor], n_quantiles, labels=False, duplicates='drop')
        for q in range(n_quantiles):
            q_ret = group[group['quantile'] == q]['daily_log_return'].mean()
            results.append({'date': date, 'quantile': q, 'return': q_ret})

    result_df = pd.DataFrame(results)
    return result_df.pivot(index='date', columns='quantile', values='return')


def plot_decile_and_longshort(group_ret, title='Decile and Long-Short Net Value'):
    """
    绘制十分位和多空净值图
    """
    nav_df = (1 + group_ret).cumprod()
    long_short_nav = (1 + (group_ret[9] - group_ret[0])).cumprod()

    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-white')  # Set white background style
    for col in nav_df.columns:
        plt.plot(nav_df.index, nav_df[col], label=f'Q{col}')
    plt.plot(long_short_nav.index, long_short_nav, label='Long-Short (Q9-Q0)', color='black', linewidth=2, linestyle='--')
    plt.title(title)
    plt.ylabel('Net Value')
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def compare_with_index(group_ret, hs300_ret):
    """
    与沪深300指数比较
    """
    long_short = group_ret[9] - group_ret[0]
    nav_factor = (1 + long_short).cumprod().rename('Factor Long-Short Portfolio')
    df = pd.concat([nav_factor, hs300_ret['hs300_return']], axis=1).dropna()
    df['hs300_nav'] = (1 + df['hs300_return']).cumprod()
    df[['Factor Long-Short Portfolio', 'hs300_nav']].plot(figsize=(10, 5), title='Factor Long-Short vs HS300')
    plt.style.use('seaborn-white')  # Set white background style
    plt.ylabel('Net Value')
    plt.show()


def evaluate_factors(data, factor_list, reverse_factors=None, hs300_ret=None):
    """
    评估因子表现
    """
    if reverse_factors is None:
        reverse_factors = []

    results = []
    for factor in factor_list:
        print(f"\nProcessing Factor: {factor}")
        df_ic = data.copy()
        df_group = data.copy()

        # Calculate IC using the original factor
        rank_ic, rank_icir = calc_rank_ic(df_ic, factor)

        # If it's a reverse factor, only used for sorting and net value
        if factor in reverse_factors:
            df_group[factor] = -df_group[factor]

        group_ret = factor_group_returns(df_group, factor)
        long_short_ret = group_ret[9] - group_ret[0]
        ann_ret, max_dd = calc_annual_return_max_drawdown(long_short_ret)

        results.append({
            'Factor': factor,
            'RankIC Mean': rank_ic.mean(),
            'RankICIR': rank_icir,
            'Annual Return': ann_ret,
            'Max Drawdown': max_dd
        })

        # Visualization
        plot_decile_and_longshort(group_ret, title=f'{factor} Decile and Long-Short Net Value Plot')
        if hs300_ret is not None:
            compare_with_index(group_ret, hs300_ret)

    return pd.DataFrame(results)


def main_analysis():
    """
    主分析流程
    """
    print("开始加载数据...")
    df = load_data()
    print("数据加载完成")
    
    print("计算AMBE相关性...")
    correlation_results_ambe = calculate_correlations(df, 'AMBE')
    print(correlation_results_ambe)
    
    print("计算AMBMP相关性...")
    correlation_results_ambmp = calculate_correlations(df, 'AMBMP')
    print(correlation_results_ambmp)
    
    print("计算VIF...")
    vif_data = calculate_vif(df)
    print(vif_data)
    
    print("准备市场数据...")
    df = prepare_market_data(df)
    
    print("准备面板数据...")
    panel_df, variables = prepare_panel_data(df)
    
    print("计算标准化后VIF...")
    df_vif = calculate_vif_after_standardization(panel_df, variables)
    print("VIF after standardization:")
    print(df_vif.to_string(index=False))
    
    print("运行基准回归...")
    model_ambe, model_ambmp = run_baseline_regression(panel_df)
    print("AMBE模型结果:")
    print(model_ambe)
    print("AMBMP模型结果:")
    print(model_ambmp)
    
    print("运行分市场回归...")
    market_model_ambe, market_model_ambmp = run_market_regression(panel_df)
    print("分市场AMBMP模型结果:")
    print(market_model_ambmp)
    
    print("清理面板数据中的无穷大和缺失值...")
    panel_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel_df.dropna(subset=['AMBMP', 'RV', 'Skewness', 'Kurtosis', 'Turnover_Rate', 'bull_market', 'bear_market'], inplace=True)
    
    print("运行RV分组回归...")
    low_rv_ambe_model, low_rv_ambmp_model, high_rv_ambe_model, high_rv_ambmp_model = run_rv_group_regression(panel_df)
    print("低RV组 AMBMP 回归结果：")
    print(low_rv_ambmp_model)
    print("高RV组 AMBMP 回归结果：")
    print(high_rv_ambmp_model)
    
    print("运行plevel分组回归...")
    model_low_ambe, model_low_ambmp, model_high_ambe, model_high_ambmp = run_plevel_group_regression(panel_df)
    print("低plevel AMBMP模型结果:")
    print(model_low_ambmp)
    print("高plevel AMBMP模型结果:")
    print(model_high_ambmp)
    
    print("运行五组分组回归...")
    ambe_df, ambmp_df = run_quantile_regression(df)
    print("AMBE 分组回归结果：")
    print(ambe_df)
    print("\nAMBMP 分组回归结果：")
    print(ambmp_df)
    
    print("运行Granger因果检验...")
    granger_df = run_granger_causality_test(df)
    print(granger_df)
    
    print("加载回测数据...")
    backtest_data = load_backtest_data()
    print("回测数据列名:")
    print(backtest_data.columns)
    
    print("加载沪深300数据...")
    hs300 = load_hs300_data()
    
    print("进行因子评估...")
    result_df = evaluate_factors(backtest_data, ['AMBE', 'AMBMP'], reverse_factors=['AMBE'], hs300_ret=hs300)
    print(result_df)
    
    return {
        'correlation_ambe': correlation_results_ambe,
        'correlation_ambmp': correlation_results_ambmp,
        'vif_data': vif_data,
        'model_ambe': model_ambe,
        'model_ambmp': model_ambmp,
        'market_model_ambmp': market_model_ambmp,
        'low_rv_ambmp_model': low_rv_ambmp_model,
        'high_rv_ambmp_model': high_rv_ambmp_model,
        'model_low_ambmp': model_low_ambmp,
        'model_high_ambmp': model_high_ambmp,
        'ambe_quantile_results': ambe_df,
        'ambmp_quantile_results': ambmp_df,
        'granger_results': granger_df,
        'factor_evaluation': result_df
    }


if __name__ == "__main__":
    results = main_analysis()
    print("分析完成！")