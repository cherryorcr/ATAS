import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import os

# ================= 配置区 =================
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 辅助函数：智能读取文件
# =========================================================
def read_file_smartly(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        try:
            return pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            print(f"⚠️ UTF-8读取失败，尝试使用 GBK 编码读取 {os.path.basename(file_path)}...")
            return pd.read_csv(file_path, encoding='gbk')
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


# =========================================================
# 1. 读取项目明细表 (关键修改：只取第三级名称)
# =========================================================
def load_project_data(file_path):
    print(f"--- [1/4] 正在读取项目表: {os.path.basename(file_path)} ---")
    try:
        df = read_file_smartly(file_path)
    except Exception as e:
        print(f"❌ 项目表读取失败: {e}")
        return pd.DataFrame()

    # 模糊匹配列名
    df.columns = df.columns.str.strip()

    # 检查列
    melt_cols = ['AI匹配技术_1', 'AI匹配技术_2', 'AI匹配技术_3']
    existing_melt_cols = [c for c in melt_cols if c in df.columns]

    if not existing_melt_cols:
        print(f"❌ 错误: 未找到技术列 {melt_cols}，请检查表头。")
        return pd.DataFrame()

    # 转换时间
    df['Start_Time_Extracted'] = pd.to_datetime(df['Start_Time_Extracted'], errors='coerce')

    print("   > 正在合并技术列 (Melt)...")
    df_melted = df.melt(
        id_vars=['Start_Time_Extracted'],
        value_vars=existing_melt_cols,
        value_name='Technology'
    ).dropna(subset=['Technology', 'Start_Time_Extracted'])

    # === 关键修改开始 ===
    # 1. 转为字符串并去空格
    df_melted['Technology'] = df_melted['Technology'].astype(str).str.strip()

    # 2. 【核心】只保留最后一个横杠后的内容
    # 例如："先进制造-工业...-现场总线技术" -> 变成 "现场总线技术"
    print("   > 正在处理技术名称 (截取 '-' 后的第三级)...")
    df_melted['Technology'] = df_melted['Technology'].apply(lambda x: x.split('-')[-1].strip())
    # === 关键修改结束 ===

    # 过滤无效字符
    df_melted = df_melted[~df_melted['Technology'].isin(['nan', '无', '', 'None'])]

    print(f"   > 项目表加载完成: 共 {len(df_melted)} 条有效技术记录")
    # 打印前几个看看对不对
    print(f"   > 名称示例: {df_melted['Technology'].head(3).values}")

    return df_melted


# =========================================================
# 2. 读取共现权重表
# =========================================================
def load_cooc_data(file_path, weight_threshold=10):
    print(f"--- [2/4] 正在读取共现表: {os.path.basename(file_path)} ---")
    try:
        df = read_file_smartly(file_path)
    except Exception as e:
        print(f"❌ 共现表读取失败: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()

    if not {'Source', 'Target', 'Weight'}.issubset(df.columns):
        print("❌ 错误: 共现表缺少 Source/Target/Weight 列")
        return pd.DataFrame()

    initial_count = len(df)
    df_filtered = df[df['Weight'] > weight_threshold].copy()

    candidates = df_filtered[['Source', 'Target', 'Weight']].copy()
    candidates['Source'] = candidates['Source'].astype(str).str.strip()
    candidates['Target'] = candidates['Target'].astype(str).str.strip()

    print(f"   > 筛选后剩余: {len(candidates)} 条 (原 {initial_count} 条)")
    return candidates


# =========================================================
# 3. 准备时间序列矩阵
# =========================================================
def prepare_time_series(df_long, freq='M'):
    print(f"--- [3/4] 构建时间序列 (粒度: {freq}) ---")
    df = df_long.set_index('Start_Time_Extracted').copy()

    # 统计
    ts_counts = df.groupby([pd.Grouper(freq=freq), 'Technology']).size().unstack(fill_value=0)

    # 归一化
    period_totals = ts_counts.sum(axis=1)
    period_totals[period_totals == 0] = 1
    ts_freq = ts_counts.div(period_totals, axis=0).fillna(0)

    print(f"   > 时间序列矩阵构建完成，包含 {ts_freq.shape[1]} 个技术")
    return ts_freq


# =========================================================
# 4. 核心分析
# =========================================================
def analyze_tech_relations(ts_data, candidates, max_lag=12):
    print("--- [4/4] 开始分析技术上下游关系 ---")
    results = []
    available_techs = set(ts_data.columns)

    skipped = 0
    calculated = 0

    for idx, row in candidates.iterrows():
        tech_a = row['Source']
        tech_b = row['Target']
        weight = row['Weight']

        if tech_a not in available_techs or tech_b not in available_techs:
            skipped += 1
            continue

        seq_a = ts_data[tech_a]
        seq_b = ts_data[tech_b]

        if len(seq_a) < 6: continue

        # 计算互相关
        a_std = (seq_a - seq_a.mean()) / (seq_a.std() + 1e-9)
        b_std = (seq_b - seq_b.mean()) / (seq_b.std() + 1e-9)
        n = len(a_std)
        ccf = np.correlate(a_std, b_std, mode='full') / n
        lags = np.arange(-n + 1, n)

        mask = (lags >= -max_lag) & (lags <= max_lag)
        best_idx = np.argmax(np.abs(ccf[mask]))
        best_lag = lags[mask][best_idx]
        max_corr = ccf[mask][best_idx]

        direction = "同步/不确定"
        if abs(max_corr) < 0.2:
            direction = "弱相关"
        elif best_lag > 0:
            direction = f"{tech_a} -> {tech_b}"
        elif best_lag < 0:
            direction = f"{tech_b} -> {tech_a}"

        p_val = None
        if len(seq_a) > 15:
            try:
                gc = grangercausalitytests(pd.DataFrame({tech_b: seq_b, tech_a: seq_a}), [1], verbose=False)
                p_val = gc[1][0]['ssr_ftest'][1]
            except:
                pass

        results.append({
            'Source': tech_a, 'Target': tech_b, 'Weight': weight,
            'Max_Corr': round(max_corr, 3), 'Lag': best_lag,
            'Direction': direction, 'Granger_P': round(p_val, 4) if p_val else None
        })
        calculated += 1

    print(f"   > 分析完成: 成功计算 {calculated} 对，跳过 {skipped} 对 (名称未匹配)")
    return pd.DataFrame(results)


# =========================================================
# 可视化
# =========================================================
def plot_result_pair(ts_data, tech_a, tech_b, lag):
    plt.figure(figsize=(12, 4))
    d1 = ts_data[tech_a]
    d2 = ts_data[tech_b]
    plt.plot(d1.index, (d1 - d1.min()) / (d1.max() - d1.min()), label=tech_a)
    plt.plot(d2.index, (d2 - d2.min()) / (d2.max() - d2.min()), label=tech_b, linestyle='--')
    plt.title(f"{tech_a} vs {tech_b} (Lag: {lag})")
    plt.legend()
    plt.show()


# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":
    # 路径配置
    project_file = r"D:\predict\0.1\data\2025_Project_Flattened_Report_FullPath.csv"
    cooc_file = r"D:\predict\0.1\data\2025_External_Tech_Weighted_Graph.csv"

    # 1. 加载
    df_proj_long = load_project_data(project_file)
    df_candidates = load_cooc_data(cooc_file, weight_threshold=10)

    # 2. 分析
    if not df_proj_long.empty and not df_candidates.empty:
        ts_matrix = prepare_time_series(df_proj_long, freq='M')
        final_df = analyze_tech_relations(ts_matrix, df_candidates, max_lag=12)

        if not final_df.empty:
            final_df['Abs_Corr'] = final_df['Max_Corr'].abs()
            final_df = final_df.sort_values('Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])

            print("\n=== Top 10 结果 ===")
            print(final_df.head(10))

            final_df.to_csv("最终上下游分析结果2025.csv", index=False, encoding='utf-8-sig')

            top_row = final_df.iloc[0]
            plot_result_pair(ts_matrix, top_row['Source'], top_row['Target'], top_row['Lag'])
        else:
            print("\n❌ 依然没有结果？")
            print("请检查：截取后的项目表技术名，是否真的和共现表里的名字一模一样？(有无空格/括号差异)")